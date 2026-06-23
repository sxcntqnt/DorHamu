import geopandas as gpd
import pandas as pd
from google.cloud import logging
from google.cloud import storage
from zenml import step

@step
def perform_spatial_joins(unified_df: gpd.GeoDataFrame, african_gdf: gpd.GeoDataFrame, osm_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Perform spatial joins and add spatial features to unified dataset."""
    # Initialize logging
    logger = logging.Client().logger("matatu_pipeline")

    # Load unified dataset from input (already provided as argument)
    try:
        logger.log_text("Received unified dataset for spatial joins")
    except Exception as e:
        logger.log_text(f"Unified dataset processing error: {str(e)}")
        raise

    # Ensure consistent CRS
    try:
        unified_df = unified_df.to_crs("EPSG:4326")
        african_gdf = african_gdf.to_crs("EPSG:4326")
        osm_gdf = osm_gdf.to_crs("EPSG:4326")
        logger.log_text("Aligned CRS for all datasets")
    except Exception as e:
        logger.log_text(f"CRS alignment error: {str(e)}")
        raise

    # Spatial join H3 hexagons with African boundaries
    try:
        h3_countries = gpd.sjoin(
            unified_df[["hex_code", "geometry", "road_id", "edge_id", "passenger_demand"]],
            african_gdf[["country_name", "geometry"]],
            how="left",
            predicate="intersects"
        )
        h3_countries["country_name"] = h3_countries["country_name"].fillna("Unknown")
        logger.log_text("Performed H3-countries spatial join")
    except Exception as e:
        logger.log_text(f"H3-countries spatial join error: {str(e)}")
        raise

    # Spatial join with OSM regions
    try:
        unified_df = gpd.sjoin(
            h3_countries,
            osm_gdf[["region_id", "geometry"]],
            how="left",
            predicate="intersects"
        )
        unified_df["region_id"] = unified_df["region_id"].fillna(-1)
        logger.log_text("Performed H3-OSM spatial join")
    except Exception as e:
        logger.log_text(f"H3-OSM spatial join error: {str(e)}")
        raise

    # Add spatial features
    try:
        unified_df["hex_area"] = unified_df.geometry.area
        unified_df["hex_centroid_x"] = unified_df.geometry.centroid.x
        unified_df["hex_centroid_y"] = unified_df.geometry.centroid.y
        unified_df["urban_overlap"] = unified_df.geometry.intersection(osm_gdf.geometry.unary_union).area / unified_df["hex_area"]
        unified_df["urban_overlap"] = unified_df["urban_overlap"].fillna(0)
        logger.log_text("Added spatial features: area, centroid, urban overlap")
    except Exception as e:
        logger.log_text(f"Spatial features error: {str(e)}")
        raise

    # Save updated unified dataset
    try:
        gcs_client = storage.Client(project="my_project")
        bucket = gcs_client.bucket("my_bucket")
        unified_df.to_parquet("/tmp/unified_df_updated.parquet")
        bucket.blob("processed_data/unified_df_updated.parquet").upload_from_filename("/tmp/unified_df_updated.parquet")
        logger.log_text("Saved updated unified dataset to GCS")
    except Exception as e:
        logger.log_text(f"Unified dataset save error: {str(e)}")
        raise

    return unified_df
