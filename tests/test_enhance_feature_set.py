import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from steps.enhance_feature_set import enhance_feature_set

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "hex_code": ["8a1e3c2f4a7ffff", "8a1e3c2f4a8ffff"],
        "passenger_demand": [10, 20],
        "hex_centroid_x": [0.5, 2.5],
        "hex_centroid_y": [0.5, 2.5]
    })

@pytest.fixture
def mock_logging_client():
    with patch("google.cloud.logging.Client") as mock_client:
        mock_logger = MagicMock()
        mock_client.return_value.logger.return_value = mock_logger
        yield mock_logger

def test_enhance_feature_set_success(sample_df, mock_logging_client):
    mock_logger = mock_logging_client
    enhanced_df = enhance_feature_set(sample_df, enhance_features=True)
    
    assert isinstance(enhanced_df, pd.DataFrame)
    assert "passenger_demand_normalized" in enhanced_df.columns
    assert "hour_12" in enhanced_df.columns
    assert len(enhanced_df) == 2
    mock_logger.log_text.assert_called_with("Enhanced feature set")

def test_enhance_feature_set_no_enhance(sample_df, mock_logging_client):
    mock_logger = mock_logging_client
    enhanced_df = enhance_feature_set(sample_df, enhance_features=False)
    
    assert isinstance(enhanced_df, pd.DataFrame)
    assert enhanced_df.equals(sample_df)
    mock_logger.log_text.assert_called_with("Skipped feature enhancement")

def test_enhance_feature_set_empty_input(mock_logging_client):
    mock_logger = mock_logging_client
    empty_df = pd.DataFrame()
    enhanced_df = enhance_feature_set(empty_df, enhance_features=True)
    
    assert isinstance(enhanced_df, pd.DataFrame)
    assert enhanced_df.empty
    mock_logger.log_text.assert_called_with("Enhanced feature set")

def test_enhance_feature_set_missing_columns(mock_logging_client):
    mock_logger = mock_logging_client
    invalid_df = pd.DataFrame({"id": [1, 2]})
    with pytest.raises(KeyError):
        enhance_feature_set(invalid_df, enhance_features=True)
    mock_logger.log_text.assert_called_with("Missing required columns: KeyError")
