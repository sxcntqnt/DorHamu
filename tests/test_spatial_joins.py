import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from unittest.mock import patch, MagicMock
from steps.spatial_joins import perform_spatial_joins

@pytest.fixture
def sample_gdf():
    return gpd.GeoDataFrame(
        {"id": [1, 2], "geometry": [Polygon([(0, 0), (1, 0), (1, 1)]), Polygon([(2, 2), (3, 2), (3, 3)])]},
        crs="EPSG:4326"
    )

@pytest.fixture
def sample_h3_gdf():
    return gpd.GeoDataFrame(
        {"hex_code": ["8a1e3c2f4a7ffff", "8a1e3c2f4a8ffff"], "geometry": [Point(0.5, 0.5), Point(2.5, 2.5)]},
        crs="EPSG:4326"
    )

@pytest.fixture
def mock_logging_client():
    with patch("google.cloud.logging.Client") as mock_client:
        mock_logger = MagicMock()
        mock_client.return_value.logger.return_value = mock_logger
        yield mock_logger

def test_perform_spatial_joins_success(sample_gdf, sample_h3_gdf, mock_logging_client):
    mock_logger = mock_logging_client
    with patch("geopandas.sjoin", return_value=pd.DataFrame({
        "hex_code": ["8a1e3c2f4a7ffff"], "id_left": [1], "id_right": [1]
    })):
        unified_df = perform_spatial_joins(sample_gdf, sample_gdf, sample_h3_gdf)
    
    assert isinstance(unified_df, pd.DataFrame)
    assert "hex_code" in unified_df.columns
    assert len(unified_df) == 1
    mock_logger.log_text.assert_called_with("Performed spatial joins")

def test_perform_spatial_joins_empty_input(mock_logging_client):
    mock_logger = mock_logging_client
    empty_gdf = gpd.GeoDataFrame({"id": [], "geometry": []}, crs="EPSG:4326")
    empty_h3_gdf = gpd.GeoDataFrame({"hex_code": [], "geometry": []}, crs="EPSG:4326")
    with patch("geopandas.sjoin", return_value=pd.DataFrame()):
        unified_df = perform_spatial_joins(empty_gdf, empty_gdf, empty_h3_gdf)
    
    assert isinstance(unified_df, pd.DataFrame)
    assert unified_df.empty
    mock_logger.log_text.assert_called_with("Performed spatial joins")

def test_perform_spatial_joins_sjoin_error(mock_logging_client, sample_gdf, sample_h3_gdf):
    mock_logger = mock_logging_client
    with patch("geopandas.sjoin", side_effect=Exception("Spatial join error")):
        with pytest.raises(Exception, match="Spatial join error"):
            perform_spatial_joins(sample_gdf, sample_gdf, sample_h3_gdf)
    mock_logger.log_text.assert_called_with("Spatial join error: Spatial join error")
