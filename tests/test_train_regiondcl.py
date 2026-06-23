import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from steps.train_regiondcl import train_regiondcl_model

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "hex_code": ["8a1e3c2f4a7ffff", "8a1e3c2f4a8ffff"],
        "hex_centroid_x": [0.5, 2.5],
        "hex_centroid_y": [0.5, 2.5]
    })

@pytest.fixture
def sample_pairs_df():
    return pd.DataFrame({
        "anchor_hex": ["8a1e3c2f4a7ffff"],
        "positive_hex": ["8a1e3c2f4a8ffff"],
        "negative_hex": ["8a1e3c2f4a9ffff"]
    })

@pytest.fixture
def mock_gcs_client():
    with patch("google.cloud.storage.Client") as mock_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        yield mock_client, mock_blob

@pytest.fixture
def mock_bigquery_client():
    with patch("google.cloud.bigquery.Client") as mock_client:
        yield mock_client

@pytest.fixture
def mock_logging_client():
    with patch("google.cloud.logging.Client") as mock_client:
        mock_logger = MagicMock()
        mock_client.return_value.logger.return_value = mock_logger
        yield mock_logger

def test_train_regiondcl_model_success(sample_df, sample_pairs_df, mock_gcs_client, mock_bigquery_client, mock_logging_client):
    mock_client, mock_blob = mock_gcs_client
    mock_bq_client = mock_bigquery_client
    mock_logger = mock_logging_client
    mock_raster_data = np.zeros((3, 64, 64))

    with patch("rasterio.open", MagicMock()) as mock_raster:
        mock_raster.return_value.__enter__.return_value.read.return_value = mock_raster_data
        with patch("tensorflow.keras.Model") as mock_model:
            mock_model_instance = MagicMock()
            mock_model.return_value = mock_model_instance
            mock_model_instance.predict.return_value = np.array([[1.0, 2.0]])
            with patch("bigquery.Client.query", return_value=MagicMock(to_dataframe=MagicMock(return_value=sample_pairs_df))):
                h3_embeddings_df, osm_embeddings_df = train_regiondcl_model(
                    unified_df=sample_df,
                    raster_data_path="gs://my_bucket/input_data/nairobi_raster.tif",
                    pairs_query="SELECT * FROM contrastive_pairs"
                )
    
    assert isinstance(h3_embeddings_df, pd.DataFrame)
    assert isinstance(osm_embeddings_df, pd.DataFrame)
    assert "hex_code" in h3_embeddings_df.columns
    assert "embedding" in h3_embeddings_df.columns
    mock_logger.log_text.assert_called_with("Stored RegionDCL embeddings in GCS")

def test_train_regiondcl_model_raster_error(sample_df, sample_pairs_df, mock_gcs_client, mock_bigquery_client, mock_logging_client):
    mock_client, mock_blob = mock_gcs_client
    mock_bq_client = mock_bigquery_client
    mock_logger = mock_logging_client
    with patch("rasterio.open", side_effect=Exception("Raster error")):
        with pytest.raises(Exception, match="Raster data load error"):
            train_regiondcl_model(
                unified_df=sample_df,
                raster_data_path="gs://my_bucket/input_data/nairobi_raster.tif",
                pairs_query="SELECT * FROM contrastive_pairs"
            )
    mock_logger.log_text.assert_called_with("Raster data load error: Raster error")
