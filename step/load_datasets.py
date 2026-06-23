import geopandas as gpd
import pandas as pd
import bigframes.pandas as bpd
import h3
from google.cloud import bigquery
from google.cloud import storage
from google.cloud import logging
from shapely.geometry import Polygon
from pathlib import Path
from zenml import step
from typing import Tuple

@step
def load_datasets() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    """Load and process african_boundaries, osm_regions, H3 indices, and unified dataset with road mappings."""
    # Initialize logging
    logging_client = logging.Client()
    logger = logging_client.logger("matatu_pipeline")

    # Set up BigQuery and GCS clients
    try:
        bq_client = bigquery.Client(project="my_project")
        gcs_client = storage.Client(project="my_project")
        logger.log_text("Initialized BigQuery and GCS clients")
    except Exception as e:
        logger.log_text(f"Client initialization error: {str(e)}")
        raise

    # Load geospatial datasets
    def load_gdf(file_path, file_name):
        try:
            if file_path.startswith("gs://"):
                bucket_name, blob_name = file_path.replace("gs://", "").split("/", 1)
                bucket = gcs_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                local_path = f"/tmp/{blob_name}"
                blob.download_to_filename(local_path)
                gdf = gpd.read_parquet(local_path)
            else:
                gdf = gpd.read_parquet(file_path)
            logger.log_text(f"Loaded {file_name} successfully")
            return gdf
        except Exception as e:
            logger.log_text(f"Error loading {file_name}: {str(e)}")
            raise

    african_gdf = load_gdf("gs://my_bucket/input_data/african_boundaries.parquet", "african_boundaries")
    osm_gdf = load_gdf("gs://my_bucket/input_data/osm_regions.parquet", "osm_regions")

    # Load H3 indices from BigQuery
    h3_query = """
    SELECT hex_code, resolution, country
    FROM my_project.my_dataset.h3_indices
    WHERE resolution = 8
    """
    try:
        h3_df = bpd.read_gbq(h3_query)
        logger.log_text("Loaded H3 indices from BigQuery")
    except Exception as e:
        logger.log_text(f"BigQuery H3 query error: {str(e)}")
        raise

    # Convert H3 indices to geometries
    def hex_to_polygon(hex_code):
        try:
            boundary = h3.cell_to_boundary(hex_code, geo_json=True)
            return [(lng, lat) for lat, lng in boundary]
        except Exception as e:
            logger.log_text(f"H3 boundary error for {hex_code}: {str(e)}")
            return []

    try:
        h3_geometries = h3_df["hex_code"].apply(hex_to_polygon)
        valid_geometries = [Polygon(geom) if geom else None for geom in h3_geometries]
        h3_gdf = gpd.GeoDataFrame(
            h3_df.to_pandas(),
            geometry=valid_geometries,
            crs="EPSG:4326"
        )
        h3_gdf = h3_gdf[h3_gdf.geometry.notnull()]
        logger.log_text("Converted H3 indices to geometries")
    except Exception as e:
        logger.log_text(f"H3 geometry conversion error: {str(e)}")
        raise

    # Ensure consistent CRS
    try:
        african_gdf = african_gdf.to_crs("EPSG:4326")
        osm_gdf = osm_gdf.to_crs("EPSG:4326")
        h3_gdf = h3_gdf.to_crs("EPSG:4326")
        logger.log_text("Aligned CRS for all datasets")
    except Exception as e:
        logger.log_text(f"CRS alignment error: {str(e)}")
        raise

    # Load reservation data for demand features
    reservation_query = """
    SELECT hex_code, COUNT(*) as passenger_demand, EXTRACT(HOUR FROM booking_time) as hour
    FROM my_project.my_dataset.reservations
    WHERE booking_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
    GROUP BY hex_code, hour
    """
    try:
        reservation_df = bpd.read_gbq(reservation_query)
        reservation_df = reservation_df.to_pandas()
        logger.log_text("Loaded reservation data from BigQuery")
    except Exception as e:
        logger.log_text(f"Reservation query error: {str(e)}")
        reservation_df = pd.DataFrame(columns=["hex_code", "passenger_demand", "hour"])

    # Map H3 hexagons to AStreet and SUMO road networks
    try:
        abstreet_gdf = load_gdf("gs://my_bucket/input_data/nairobi.bin.geojson", "abstreet_network")
        sumo_gdf = load_gdf("gs://my_bucket/input_data/nairobi.poly.parquet", "sumo_network")

        h3_abstreet = gpd.sjoin(h3_gdf, abstreet_gdf[["geometry", "road_id"]], how="left", predicate="intersects")
        h3_abstreet["road_id"] = h3_abstreet["road_id"].fillna(-1)

        h3_sumo = gpd.sjoin(h3_gdf, sumo_gdf[["geometry", "edge_id"]], how="left", predicate="intersects")
        h3_sumo["edge_id"] = h3_sumo["edge_id"].fillna("none")

        h3_mappings = h3_abstreet[["hex_code", "road_id"]].merge(
            h3_sumo[["hex_code", "edge_id"]], on="hex_code", how="left"
        )

        if not reservation_df.empty:
            h3_mappings = h3_mappings.merge(
                reservation_df.groupby("hex_code")["passenger_demand"].sum().reset_index(),
                on="hex_code",
                how="left"
            )
            h3_mappings["passenger_demand"] = h3_mappings["passenger_demand"].fillna(0)

        h3_mappings.to_gbq(
            "my_project.my_dataset.h3_mappings",
            if_exists="replace",
            table_schema=[
                {"name": "hex_code", "type": "STRING"},
                {"name": "road_id", "type": "FLOAT64"},
                {"name": "edge_id", "type": "STRING"},
                {"name": "passenger_demand", "type": "FLOAT64"},
            ],
        )
        logger.log_text("Generated and stored H3-to-road mappings with demand features")
    except Exception as e:
        logger.log_text(f"H3-to-road mapping error: {str(e)}")
        raise

    # Save unified dataset
    try:
        unified_df = h3_mappings.merge(h3_gdf[["hex_code", "geometry", "resolution", "country"]], on="hex_code")
        unified_df = unified_df.merge(
            reservation_df.pivot_table(
                index="hex_code", columns="hour", values="passenger_demand", fill_value=0
            ).reset_index(),
            on="hex_code",
            how="left",
        )
        unified_df.fillna(0, inplace=True)
        unified_df.to_parquet("gs://my_bucket/processed_data/unified_df.parquet")
        logger.log_text("Saved unified dataset to GCS")
    except Exception as e:
        logger.log_text(f"Unified dataset save error: {str(e)}")
        raise

    return african_gdf, osm_gdf, h3_gdf, unified_df
