import geopandas as gpd
import pandas as pd
import bigframes.pandas as bpd
from sklearn.preprocessing import StandardScaler
from google.cloud import logging
from google.cloud import storage
import h3
from zenml import step

@step
def enhance_feature_set(unified_df: gpd.GeoDataFrame) -> pd.DataFrame:
    """Enhance unified dataset with H3 hierarchy features, normalize, and store."""
    # Initialize logging
    logger = logging.Client().logger("matatu_pipeline")

    # Add H3 hierarchy features
    def add_h3_features(row):
        try:
            hex_code = row["hex_code"]
            resolution = h3.get_resolution(hex_code)
            parent_hex = h3.cell_to_parent(hex_code, resolution - 1) if resolution > 0 else None
            child_hexes = h3.cell_to_children(hex_code, resolution + 1) if resolution < 15 else []
            edge_length = h3.edge_length_m(resolution)
            return pd.Series({
                "parent_hex": parent_hex,
                "child_count": len(child_hexes),
                "hex_edge_length": edge_length
            })
        except Exception as e:
            logger.log_text(f"H3 features error for {hex_code}: {str(e)}")
            return pd.Series({"parent_hex": None, "child_count": 0, "hex_edge_length": 0})

    try:
        h3_features = unified_df.apply(add_h3_features, axis=1)
        unified_df = pd.concat([unified_df.drop(columns=["geometry"]), h3_features], axis=1)
        unified_df["urban_overlap"] = (unified_df["region_id"] > 0).astype(int)
        logger.log_text("Added H3 features and urban overlap")
    except Exception as e:
        logger.log_text(f"H3 features concatenation error: {str(e)}")
        raise

    # Normalize numerical features
    try:
        scaler = StandardScaler()
        numerical_cols = [
            "hex_area",
            "hex_centroid_x",
            "hex_centroid_y",
            "child_count",
            "hex_edge_length",
            "passenger_demand"
        ]
        # Ensure numerical columns exist and fill missing values
        for col in numerical_cols:
            if col not in unified_df.columns:
                unified_df[col] = 0
            unified_df[col] = unified_df[col].fillna(0)
        unified_df[numerical_cols] = scaler.fit_transform(unified_df[numerical_cols].astype(float))
        logger.log_text("Normalized numerical features")
    except Exception as e:
        logger.log_text(f"Feature normalization error: {str(e)}")
        raise

    # Store to BigQuery
    try:
        gcs_client = storage.Client(project="my_project")
        bucket = gcs_client.bucket("my_bucket")
        unified_bdf = bpd.DataFrame(unified_df)
        unified_bdf.to_gbq(
            "my_project.my_dataset.unified_features",
            if_exists="replace",
            table_schema=[
                {"name": "hex_code", "type": "STRING"},
                {"name": "road_id", "type": "FLOAT64"},
                {"name": "edge_id", "type": "STRING"},
                {"name": "passenger_demand", "type": "FLOAT64"},
                {"name": "country_name", "type": "STRING"},
                {"name": "region_id", "type": "FLOAT64"},
                {"name": "hex_area", "type": "FLOAT64"},
                {"name": "hex_centroid_x", "type": "FLOAT64"},
                {"name": "hex_centroid_y", "type": "FLOAT64"},
                {"name": "parent_hex", "type": "STRING"},
                {"name": "child_count", "type": "INTEGER"},
                {"name": "hex_edge_length", "type": "FLOAT64"},
                {"name": "urban_overlap", "type": "INTEGER"},
            ] + [
                {"name": f"hour_{h}", "type": "FLOAT64"} for h in range(24)
            ]
        )
        logger.log_text("Stored unified features in BigQuery")
    except Exception as e:
        logger.log_text(f"BigQuery unified features error: {str(e)}")
        raise

    # Save to GCS for downstream use
    try:
        unified_df.to_parquet("/tmp/unified_features.parquet")
        bucket.blob("processed_data/unified_features.parquet").upload_from_filename("/tmp/unified_features.parquet")
        logger.log_text("Saved unified features to GCS")
    except Exception as e:
        logger.log_text(f"Unified features save error: {str(e)}")
        raise

    return unified_df
