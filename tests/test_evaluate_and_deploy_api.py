import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from gym.spaces import Dict, Discrete, Box
from steps.evaluate_ai import evaluate_and_deploy_api
from steps.define_ma3_environment import MatatuEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

@pytest.fixture
def sample_unified_df():
    data = {
        "hex_code": ["8a1e3c2f4a7ffff", "8a1e3c2f4a8ffff"],
        "hex_centroid_x": [0.5, 2.5],
        "hex_centroid_y": [0.5, 2.5],
        "passenger_demand": [10.0, 15.0]
    }
    for h in range(24):
        data[f"hour_{h}"] = [1.0, 2.0]
    return pd.DataFrame(data)

@pytest.fixture
def mock_ppo():
    mock_policy = MagicMock(spec=PPO)
    mock_policy.predict.return_value = ({"high_level": np.array([0]), "low_level": np.array([1])}, None)
    return mock_policy

@pytest.fixture
def mock_gcs_client():
    with patch("google.cloud.storage.Client") as mock_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.return_value.bucket.return_value = mock_blob
        mock_bucket.blob.return_value = mock_blob
        yield mock_client, mock_blob

@pytest.fixture
def mock_bigquery_client():
    with patch("google.cloud.bigquery.Client") as mock_client:
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.query.return_value.to_dataframe.return_value = pd.DataFrame({
            "hex_code": ["8a1e3c2f4a7ffff"],
            "demand": [5]
        })
        yield mock_client_instance

@pytest.fixture
def mock_logging_client():
    with patch("google.cloud.logging.Client") as mock_client:
        mock_logger = MagicMock()
        mock_client.return_value.logger.return_value = mock_logger
        yield mock_logger

@pytest.fixture
def mock_monitoring_client():
    with patch("google.cloud.monitoring_v3.MetricServiceClient") as mock_client:
        yield mock_client

@pytest.fixture
def mock_matatu_env():
    env = MagicMock(spec=MatatuEnv)
    env.node_list = ["8a1e3c2f4a7ffff", "8a1e3c2f4a8ffff"]
    env.observation_space = Dict({
        "place_cells": Box(low=0, high=1, shape=(10,), dtype=np.float32),
        "hourly_demand": Box(low=0, high=10, shape=(24,), dtype=np.float32),
        "edge_congestion": Box(low=0, high=1, shape=(1,), dtype=np.float32)
    })
    env.action_space = Dict({
        "high_level": Discrete(5),
        "low_level": Discrete(3)
    })
    env.get_attr.side_effect = lambda attr, indices=None: {
        "current_node": ["8a1e3c2f4a7ffff"] * 4,
        "sumo_running": [True] * 4,
        "vehicle_id": ["veh_0"] * 4,
        "abstreet_running": [True] * 4,
        "abstreet_sim_dir": ["/tmp/abstreet"] * 4
    }[attr]
    env.reset.return_value = {
        "place_cells": np.zeros((4, 10)),
        "hourly_demand": np.ones((4, 24)),
        "edge_congestion": np.array([[0.5]] * 4)
    }
    env.step.return_value = (
        env.reset.return_value,
        np.array([10.0] * 4),
        np.array([False] * 4),
        [
            {
                "passengers": 5,
                "traffic": 0.1,
                "congestion": 0.2,
                "intrinsic_reward": 0.1,
                "her_reward": 0.5,
                "forward_error": 0.01,
                "high_level_reward": 0.3,
                "low_level_reward": 0.2,
                "current_node": "8a1e3c2f4a7ffff"
            } for _ in range(4)
        ]
    )
    env.close = MagicMock()
    return env

def test_evaluate_and_deploy_api_success(sample_unified_df, mock_ppo, mock_gcs_client, mock_bigquery_client, mock_logging_client, mock_monitoring_client, mock_matatu_env):
    mock_client, mock_blob = mock_gcs_client
    mock_bq_client = mock_bigquery_client
    mock_logger = mock_logging_client
    mock_monitoring = mock_monitoring_client

    with patch("folium.Map") as mock_map:
        with patch("folium.PolyLine") as mock_polyline:
            with patch("folium.Marker") as mock_marker:
                with patch("steps.evaluate_ai.MatatuEnv", return_value=mock_matatu_env):
                    with patch("stable_baselines3.common.vec_env.SubprocVecEnv") as mock_vec_env:
                        mock_vec_env_instance = mock_matatu_env
                        mock_vec_env.return_value = mock_vec_env_instance
                        with patch("pandas.DataFrame.to_gbq") as mock_to_gbq:
                            with patch("traci.vehicle.getFuelConsumption", return_value=1.0):
                                with patch("json.load", return_value={"bus_trips": 10, "total_trips": 100}):
                                    result = evaluate_and_deploy_api(
                                        unified_df=sample_unified_df,
                                        high_level_policy=mock_ppo,
                                        low_level_policy=mock_ppo,
                                        n_envs=4,
                                        num_eval_episodes=2
                                    )

    assert isinstance(result, dict)
    assert "evaluation_results" in result
    assert isinstance(result["evaluation_results"], pd.DataFrame)
    assert set(result["evaluation_results"].columns).issuperset({
        "reward", "passengers_served", "travel_time", "wait_time",
        "intrinsic_reward", "her_reward", "forward_error",
        "high_level_reward", "low_level_reward"
    })
    mock_to_gbq.assert_called()
    mock_logger.log_text.assert_any_call("Stored evaluation results in BigQuery")
    mock_logger.log_text.assert_any_call("Saved multi-environment route visualization to GCS")
    mock_blob.upload_from_filename.assert_any_call("/tmp/advanced_metrics.parquet")
    mock_monitoring.create_time_series.assert_called()
    mock_polyline.assert_called()
    mock_marker.assert_called()

def test_evaluate_and_deploy_api_quality_gate_failure(sample_unified_df, mock_ppo, mock_gcs_client, mock_bigquery_client, mock_logging_client, mock_monitoring_client, mock_matatu_env):
    mock_client, mock_blob = mock_gcs_client
    mock_bq_client = mock_bigquery_client
    mock_logger = mock_logging_client
    mock_monitoring = mock_monitoring_client

    # Update step to return lower metrics
    mock_matatu_env.step.return_value = (
        mock_matatu_env.reset.return_value,
        np.array([5.0] * 4),
        np.array([False] * 4),
        [
            {
                "passengers": 2,
                "traffic": 0.1,
                "congestion": 0.2,
                "intrinsic_reward": 0.05,
                "her_reward": 0.2,
                "forward_error": 0.02,
                "high_level_reward": 0.1,
                "low_level_reward": 0.1,
                "current_node": "8a1e3c2f4a7ffff"
            } for _ in range(4)
        ]
    )

    with patch("folium.Map"):
        with patch("steps.evaluate_ai.MatatuEnv", return_value=mock_matatu_env):
            with patch("stable_baselines3.common.vec_env.SubprocVecEnv") as mock_vec_env:
                mock_vec_env.return_value = mock_matatu_env
                with pytest.raises(ValueError, match="Quality gates failed"):
                    evaluate_and_deploy_api(
                        unified_df=sample_unified_df,
                        high_level_policy=mock_ppo,
                        low_level_policy=mock_ppo,
                        n_envs=4,
                        min_reward=10.0,
                        min_passengers_served=5.0,
                        min_her_reward=0.5,
                        fail_on_quality_gates=True,
                        num_eval_episodes=2
                    )
    mock_logger.log_text.assert_any_call("Evaluation error: Quality gates failed.*")

def test_evaluate_and_deploy_api_gcs_error(sample_unified_df, mock_ppo, mock_gcs_client, mock_bigquery_client, mock_logging_client, mock_monitoring_client, mock_matatu_env):
    mock_client, mock_blob = mock_gcs_client
    mock_bq_client = mock_bigquery_client
    mock_logger = mock_logging_client
    mock_monitoring = mock_monitoring_client

    mock_blob.upload_from_filename.side_effect = Exception("GCS error")

    with patch("folium.Map"):
        with patch("steps.evaluate_ai.MatatuEnv", return_value=mock_matatu_env):
            with patch("stable_baselines3.common.vec_env.SubprocVecEnv") as mock_vec_env:
                mock_vec_env.return_value = mock_matatu_env
                with pytest.raises(Exception, match="Route visualization error"):
                    evaluate_and_deploy_api(
                        unified_df=sample_unified_df,
                        high_level_policy=mock_ppo,
                        low_level_policy=mock_ppo,
                        n_envs=4,
                        num_eval_episodes=2
                    )
    mock_logger.log_text.assert_called_with("Route visualization error: GCS error")

def test_evaluate_and_deploy_api_vectorized_envs(sample_unified_df, mock_ppo, mock_gcs_client, mock_bigquery_client, mock_logging_client, mock_monitoring_client, mock_matatu_env):
    mock_client, mock_blob = mock_gcs_client
    mock_bq_client = mock_bigquery_client
    mock_logger = mock_logging_client
    mock_monitoring = mock_monitoring_client

    with patch("folium.Map"):
        with patch("steps.evaluate_ai.MatatuEnv", return_value=mock_matatu_env):
            with patch("stable_baselines3.common.vec_env.SubprocVecEnv") as mock_vec_env:
                mock_vec_env_instance = mock_matatu_env
                mock_vec_env.return_value = mock_vec_env_instance
                with patch("pandas.DataFrame.to_gbq"):
                    with patch("traci.vehicle.getFuelConsumption", return_value=1.0):
                        with patch("json.load", return_value={"bus_trips": 10, "total_trips": 100}):
                            result = evaluate_and_deploy_api(
                                unified_df=sample_unified_df,
                                high_level_policy=mock_ppo,
                                low_level_policy=mock_ppo,
                                n_envs=3,
                                num_eval_episodes=2
                            )

    assert len(mock_vec_env.call_args[0][0]) == 3  # 3 environments
    assert len(result["evaluation_results"]) == 6  # 3 envs * 2 episodes
    mock_logger.log_text.assert_any_call("Evaluated high-level and low-level policies across multiple environments")

def test_evaluate_and_deploy_api_advanced_metrics(sample_unified_df, mock_ppo, mock_gcs_client, mock_bigquery_client, mock_logging_client, mock_monitoring_client, mock_matatu_env):
    mock_client, mock_blob = mock_gcs_client
    mock_bq_client = mock_bigquery_client
    mock_logger = mock_logging_client
    mock_monitoring = mock_monitoring_client

    with patch("folium.Map"):
        with patch("steps.evaluate_ai.MatatuEnv", return_value=mock_matatu_env):
            with patch("stable_baselines3.common.vec_env.SubprocVecEnv") as mock_vec_env:
                mock_vec_env_instance = mock_matatu_env
                mock_vec_env.return_value = mock_vec_env_instance
                with patch("pandas.DataFrame.to_gbq"):
                    with patch("traci.vehicle.getFuelConsumption", return_value=1.0):
                        with patch("json.load", return_value={"bus_trips": 10, "total_trips": 100}):
                            result = evaluate_and_deploy_api(
                                unified_df=sample_unified_df,
                                high_level_policy=mock_ppo,
                                low_level_policy=mock_ppo,
                                n_envs=4,
                                num_eval_episodes=2
                            )

    eval_df = result["evaluation_results"]
    assert "intrinsic_reward" in eval_df.columns
    assert "her_reward" in eval_df.columns
    assert "forward_error" in eval_df.columns
    assert "high_level_reward" in eval_df.columns
    assert "low_level_reward" in eval_df.columns
    assert eval_df["intrinsic_reward"].mean() == pytest.approx(0.1, rel=1e-2)
    assert eval_df["her_reward"].mean() == pytest.approx(0.5, rel=1e-2)
    mock_blob.upload_from_filename.assert_any_call("/tmp/advanced_metrics.parquet")
    mock_logger.log_text.assert_any_call("Stored evaluation results and advanced metrics to GCS")

def test_evaluate_and_deploy_api_route_visualization(sample_unified_df, mock_ppo, mock_gcs_client, mock_bigquery_client, mock_logging_client, mock_monitoring_client, mock_matatu_env):
    mock_client, mock_blob = mock_gcs_client
    mock_bq_client = mock_bigquery_client
    mock_logger = mock_logging_client
    mock_monitoring = mock_monitoring_client

    with patch("folium.Map") as mock_map:
        with patch("folium.PolyLine") as mock_polyline:
            with patch("folium.Marker") as mock_marker:
                with patch("steps.evaluate_ai.MatatuEnv", return_value=mock_matatu_env):
                    with patch("stable_baselines3.common.vec_env.SubprocVecEnv") as mock_vec_env:
                        mock_vec_env_instance = mock_matatu_env
                        mock_vec_env.return_value = mock_vec_env_instance
                        with patch("pandas.DataFrame.to_gbq"):
                            with patch("traci.vehicle.getFuelConsumption", return_value=1.0):
                                with patch("json.load", return_value={"bus_trips": 10, "total_trips": 100}):
                                    evaluate_and_deploy_api(
                                        unified_df=sample_unified_df,
                                        high_level_policy=mock_ppo,
                                        low_level_policy=mock_ppo,
                                        n_envs=4,
                                        num_eval_episodes=2
                                    )

    assert mock_polyline.call_count >= 4  # One PolyLine per environment
    assert mock_marker.call_count >= 8  # Two markers (start/end) per environment
    mock_polyline.assert_called_with(
        [(0.5, 0.5)],  # Route points from unified_df
        color=pytest.any(str),  # Distinct colors
        weight=2.5,
        opacity=0.8,
        popup=pytest.any(str)
    )
    mock_logger.log_text.assert_any_call("Saved multi-environment route visualization to GCS")
