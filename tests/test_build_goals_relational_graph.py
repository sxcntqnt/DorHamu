import pytest
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from unittest.mock import patch, MagicMock
from steps.build_goals_relational_graph import build_goals_relational_graph
import subprocess

@pytest.fixture
def sample_unified_df():
    return pd.DataFrame({
        "hex_code": ["8a1e3c2f4a7ffff", "8a1e3c2f4a8ffff"],
        "hex_centroid_x": [0.5, 2.5],
        "hex_centroid_y": [0.5, 2.5]
    })

@pytest.fixture
def sample_h3_embeddings_df():
    return pd.DataFrame({
        "hex_code": ["8a1e3c2f4a7ffff", "8a1e3c2f4a8ffff"],
        "embedding": [[1.0, 2.0], [3.0, 4.0]]
    })

@pytest.fixture
def sample_osm_embeddings_df():
    return pd.DataFrame({
        "region_id": [1, 2],
        "embedding": [[5.0, 6.0], [7.0, 8.0]]
    })

@pytest.fixture
def sample_osm_gdf():
    return gpd.GeoDataFrame({
        "region_id": [1, 2],
        "geometry": [Point(0.5, 0.5), Point(2.5, 2.5)]
    }, crs="EPSG:4326")

@pytest.fixture
def mock_gcs_client():
    with patch("google.cloud.storage.Client") as mock_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        yield mock_client, mock_blob

@pytest.fixture
def mock_logging_client():
    with patch("google.cloud.logging.Client") as mock_client:
        mock_logger = MagicMock()
        mock_client.return_value.logger.return_value = mock_logger
        yield mock_logger

def test_build_goals_relational_graph_success(
    sample_unified_df,
    sample_h3_embeddings_df,
    sample_osm_embeddings_df,
    sample_osm_gdf,
    mock_gcs_client,
    mock_logging_client
):
    mock_client, mock_blob = mock_gcs_client
    mock_logger = mock_logging_client

    with patch("networkx.DiGraph") as mock_nx_graph:
        mock_nx_graph_instance = MagicMock()
        mock_nx_graph.return_value = mock_nx_graph_instance
        mock_nx_graph_instance.nodes.return_value = [
            (0, {"hex_code": "8a1e3c2f4a7ffff", "type": "hex"}),
            (1, {"hex_code": "8a1e3c2f4a8ffff", "type": "hex"}),
            (2, {"region_id": 1, "type": "region"}),
            (3, {"region_id": 2, "type": "region"})
        ]
        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.return_value = MagicMock(stdout="Cloverleaf completed")
            with patch("builtins.open") as mock_open:
                mock_open.side_effect = [
                    MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),  # Edge list write
                    MagicMock(__enter__=MagicMock(readlines=MagicMock(return_value=[
                        "0 1.0 2.0 3.0",
                        "1 4.0 5.0 6.0",
                        "2 7.0 8.0 9.0",
                        "3 10.0 11.0 12.0"
                    ])), __exit__=MagicMock())  # Embedding read
                ]
                grg_embeddings_df, hex_to_node, region_to_node = build_goals_relational_graph(
                    unified_df=sample_unified_df,
                    h3_embeddings_df=sample_h3_embeddings_df,
                    osm_embeddings_df=sample_osm_embeddings_df,
                    osm_gdf=sample_osm_gdf
                )

    assert isinstance(grg_embeddings_df, pd.DataFrame)
    assert "node_id" in grg_embeddings_df.columns
    assert "embedding" in grg_embeddings_df.columns
    assert len(grg_embeddings_df) == 4
    assert isinstance(hex_to_node, dict)
    assert isinstance(region_to_node, dict)
    assert len(hex_to_node) == 2
    assert len(region_to_node) == 2
    mock_subprocess.assert_called_with(
        [
            "cloverleaf",
            "--input", "/tmp/grg_edges.txt",
            "--output", "/tmp/grg_embeddings.txt",
            "--dims", "128",
            "--num-walks", "10000",
            "--hashes", "3",
            "--steps", "0.3",
            "--beta", "0.8"
        ],
        check=True, capture_output=True, text=True
    )
    mock_nx_graph_instance.add_node.assert_called()
    mock_nx_graph_instance.add_edge.assert_called()
    mock_blob.upload_from_filename.assert_called()
    mock_logger.log_text.assert_any_call("Wrote edge list for Cloverleaf")
    mock_logger.log_text.assert_any_call("Built Goals Relational Graph")

def test_build_goals_relational_graph_empty_input(
    mock_gcs_client,
    mock_logging_client
):
    mock_client, mock_blob = mock_gcs_client
    mock_logger = mock_logging_client
    empty_df = pd.DataFrame()
    empty_gdf = gpd.GeoDataFrame({"region_id": [], "geometry": []}, crs="EPSG:4326")

    with patch("networkx.DiGraph") as mock_nx_graph:
        mock_nx_graph_instance = MagicMock()
        mock_nx_graph.return_value = mock_nx_graph_instance
        mock_nx_graph_instance.nodes.return_value = []
        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.return_value = MagicMock(stdout="Cloverleaf completed")
            with patch("builtins.open") as mock_open:
                mock_open.side_effect = [
                    MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
                    MagicMock(__enter__=MagicMock(readlines=MagicMock(return_value=[])), __exit__=MagicMock())
                ]
                grg_embeddings_df, hex_to_node, region_to_node = build_goals_relational_graph(
                    unified_df=empty_df,
                    h3_embeddings_df=empty_df,
                    osm_embeddings_df=empty_df,
                    osm_gdf=empty_gdf
                )

    assert grg_embeddings_df.empty
    assert hex_to_node == {}
    assert region_to_node == {}
    mock_subprocess.assert_called()
    mock_logger.log_text.assert_any_call("Built Goals Relational Graph")

def test_build_goals_relational_graph_cloverleaf_error(
    sample_unified_df,
    sample_h3_embeddings_df,
    sample_osm_embeddings_df,
    sample_osm_gdf,
    mock_gcs_client,
    mock_logging_client
):
    mock_client, mock_blob = mock_gcs_client
    mock_logger = mock_logging_client

    with patch("networkx.DiGraph") as mock_nx_graph:
        mock_nx_graph_instance = MagicMock()
        mock_nx_graph.return_value = mock_nx_graph_instance
        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.side_effect = subprocess.CalledProcessError(
                returncode=1, cmd=["cloverleaf"], stderr="Cloverleaf failed"
            )
            with patch("builtins.open") as mock_open:
                mock_open.side_effect = [
                    MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
                ]
                with pytest.raises(subprocess.CalledProcessError, match="Cloverleaf failed"):
                    build_goals_relational_graph(
                        unified_df=sample_unified_df,
                        h3_embeddings_df=sample_h3_embeddings_df,
                        osm_embeddings_df=sample_osm_embeddings_df,
                        osm_gdf=sample_osm_gdf
                    )

    mock_logger.log_text.assert_any_call("Cloverleaf execution error: Cloverleaf failed")
