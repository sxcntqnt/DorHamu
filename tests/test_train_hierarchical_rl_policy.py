import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from gym.spaces import Dict, Discrete, Box
from steps.train_rl_policy import train_hierarchical_rl_policy
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
def mock_logging_client():
    with patch("google.cloud.logging.Client") as mock_client:
        mock_logger = MagicMock()
        mock_client.return_value.logger.return_value = mock_logger
        yield mock_logger

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
    with patch("bigframes.pandas.read_gbq") as mock_read_gbq:
        yield mock_read_gbq

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
    env.curriculum_phase = 0
    env.forward_error = 0.0
    env.desired_goal = np.array([1.0, 2.0])
    env.achieved_goal = np.array([0.5, 1.5])
    env.reset.return_value = {
        "place_cells": np.zeros(10),
        "hourly_demand": np.ones(24),
        "edge_congestion": np.array([0.5])
    }
    env.step.return_value = (
        env.reset.return_value,
        1.0,
        False,
        {
            "passengers": 2,
            "traffic": 0.1,
            "congestion": 0.2,
            "her_reward": 0.5,
            "intrinsic_reward": 0.1,
            "forward_error": 0.01,
            "high_level_reward": 0.3,
            "low_level_reward": 0.2,
            "desired_goal": env.desired_goal,
            "achieved_goal": env.achieved_goal,
            "current_node": "8a1e3c2f4a7ffff"
        }
    )
    env.set_forward_error = MagicMock()
    env.close = MagicMock()
    return env

def test_train_hierarchical_rl_policy_success(sample_unified_df, mock_logging_client, mock_gcs_client, mock_bigquery_client, mock_matatu_env):
    mock_logger = mock_logging_client
    mock_client, mock_blob = mock_gcs_client
    mock_read_gbq = mock_bigquery_client

    # Mock data loading
    mock_read_gbq.side_effect = [
        pd.DataFrame({"hex_code": ["8a1e3c2f4a7ffff"], "road_id": [1], "edge_id": [1], "passenger_demand": [10.0]}),
        pd.DataFrame({"hex_code": ["8a1e3c2f4a7ffff"], "embedding": [[0.1, 0.2]], "type": ["node"]})
    ]
    mock_blob.download_to_filename.return_value = None
    sample_unified_df.to_parquet("/tmp/unified_features.parquet")

    with patch("steps.train_rl_policy.MatatuEnv", return_value=mock_matatu_env):
        with patch("stable_baselines3.common.vec_env.SubprocVecEnv") as mock_vec_env:
            mock_vec_env_instance = MagicMock()
            mock_vec_env.return_value = mock_vec_env_instance
            with patch("stable_baselines3.PPO") as mock_ppo:
                mock_ppo_instance = MagicMock()
                mock_ppo.return_value = mock_ppo_instance
                with patch("torch.nn.MSELoss", return_value=MagicMock(item=lambda: 0.1)):
                    high_level_policy, low_level_policy = train_hierarchical_rl_policy(sample_unified_df, n_envs=4)

    assert isinstance(high_level_policy, PPO)
    assert isinstance(low_level_policy, PPO)
    mock_vec_env.assert_called()
    mock_ppo.assert_called()
    mock_logger.log_text.assert_any_call("Trained and saved high-level policy to GCS")
    mock_logger.log_text.assert_any_call("Trained and saved low-level policy to GCS")
    mock_blob.upload_from_filename.assert_called()
    # Check policy configurations
    high_level_call = mock_ppo.call_args_list[0][1]
    assert high_level_call["n_steps"] == 2048
    assert high_level_call["policy_kwargs"]["net_arch"] == [dict(pi=[512, 256], vf=[512, 256])]
    low_level_call = mock_ppo.call_args_list[1][1]
    assert low_level_call["n_steps"] == 1024
    assert low_level_call["policy_kwargs"]["net_arch"] == [dict(pi=[256, 128], vf=[256, 128])]

def test_train_hierarchical_rl_policy_empty_input(mock_logging_client, mock_gcs_client, mock_bigquery_client, mock_matatu_env):
    mock_logger = mock_logging_client
    mock_client, mock_blob = mock_gcs_client
    mock_read_gbq = mock_bigquery_client
    empty_df = pd.DataFrame()

    # Mock data loading with empty results
    mock_read_gbq.side_effect = [
        pd.DataFrame({"hex_code": [], "road_id": [], "edge_id": [], "passenger_demand": []}),
        pd.DataFrame({"hex_code": [], "embedding": [], "type": []})
    ]
    mock_blob.download_to_filename.return_value = None
    empty_df.to_parquet("/tmp/unified_features.parquet")

    with patch("steps.train_rl_policy.MatatuEnv", return_value=mock_matatu_env):
        with patch("stable_baselines3.common.vec_env.SubprocVecEnv") as mock_vec_env:
            mock_vec_env_instance = MagicMock()
            mock_vec_env.return_value = mock_vec_env_instance
            with patch("stable_baselines3.PPO") as mock_ppo:
                mock_ppo_instance = MagicMock()
                mock_ppo.return_value = mock_ppo_instance
                with patch("torch.nn.MSELoss", return_value=MagicMock(item=lambda: 0.1)):
                    high_level_policy, low_level_policy = train_hierarchical_rl_policy(empty_df, n_envs=2)

    assert isinstance(high_level_policy, PPO)
    assert isinstance(low_level_policy, PPO)
    mock_logger.log_text.assert_any_call("Trained and saved high-level policy to GCS")
    mock_logger.log_text.assert_any_call("Trained and saved low-level policy to GCS")
    mock_vec_env.assert_called()

def test_train_hierarchical_rl_policy_vectorized_envs(sample_unified_df, mock_logging_client, mock_gcs_client, mock_bigquery_client, mock_matatu_env):
    mock_logger = mock_logging_client
    mock_client, mock_blob = mock_gcs_client
    mock_read_gbq = mock_bigquery_client

    mock_read_gbq.side_effect = [
        pd.DataFrame({"hex_code": ["8a1e3c2f4a7ffff"], "road_id": [1], "edge_id": [1], "passenger_demand": [10.0]}),
        pd.DataFrame({"hex_code": ["8a1e3c2f4a7ffff"], "embedding": [[0.1, 0.2]], "type": ["node"]})
    ]
    mock_blob.download_to_filename.return_value = None
    sample_unified_df.to_parquet("/tmp/unified_features.parquet")

    with patch("steps.train_rl_policy.MatatuEnv", return_value=mock_matatu_env):
        with patch("stable_baselines3.common.vec_env.SubprocVecEnv") as mock_vec_env:
            mock_vec_env_instance = MagicMock()
            mock_vec_env.return_value = mock_vec_env_instance
            with patch("stable_baselines3.PPO") as mock_ppo:
                mock_ppo_instance = MagicMock()
                mock_ppo.return_value = mock_ppo_instance
                with patch("torch.nn.MSELoss", return_value=MagicMock(item=lambda: 0.1)):
                    train_hierarchical_rl_policy(sample_unified_df, n_envs=3)

    # Check SubprocVecEnv was called with correct number of environments
    assert len(mock_vec_env.call_args[0][0]) == 3  # 3 environments
    mock_logger.log_text.assert_any_call("Created MatatuEnv with curriculum phase 0")

def test_train_hierarchical_rl_policy_curriculum_phases(sample_unified_df, mock_logging_client, mock_gcs_client, mock_bigquery_client, mock_matatu_env):
    mock_logger = mock_logging_client
    mock_client, mock_blob = mock_gcs_client
    mock_read_gbq = mock_bigquery_client

    mock_read_gbq.side_effect = [
        pd.DataFrame({"hex_code": ["8a1e3c2f4a7ffff"], "road_id": [1], "edge_id": [1], "passenger_demand": [10.0]}),
        pd.DataFrame({"hex_code": ["8a1e3c2f4a7ffff"], "embedding": [[0.1, 0.2]], "type": ["node"]})
    ]
    mock_blob.download_to_filename.return_value = None
    sample_unified_df.to_parquet("/tmp/unified_features.parquet")

    with patch("steps.train_rl_policy.MatatuEnv", return_value=mock_matatu_env):
        with patch("stable_baselines3.common.vec_env.SubprocVecEnv") as mock_vec_env:
            mock_vec_env_instance = MagicMock()
            mock_vec_env.return_value = mock_vec_env_instance
            with patch("stable_baselines3.PPO") as mock_ppo:
                mock_ppo_instance = MagicMock()
                mock_ppo.return_value = mock_ppo_instance
                with patch("torch.nn.MSELoss", return_value=MagicMock(item=lambda: 0.1)):
                    train_hierarchical_rl_policy(sample_unified_df, n_envs=4)

    # Check logging for curriculum phases
    mock_logger.log_text.assert_any_call("High-level policy: Starting curriculum phase 0")
    mock_logger.log_text.assert_any_call("High-level policy: Starting curriculum phase 1")
    mock_logger.log_text.assert_any_call("High-level policy: Starting curriculum phase 2")
    mock_logger.log_text.assert_any_call("Low-level policy: Starting curriculum phase 0")

def test_train_hierarchical_rl_policy_advanced_metrics(sample_unified_df, mock_logging_client, mock_gcs_client, mock_bigquery_client, mock_matatu_env):
    mock_logger = mock_logging_client
    mock_client, mock_blob = mock_gcs_client
    mock_read_gbq = mock_bigquery_client

    mock_read_gbq.side_effect = [
        pd.DataFrame({"hex_code": ["8a1e3c2f4a7ffff"], "road_id": [1], "edge_id": [1], "passenger_demand": [10.0]}),
        pd.DataFrame({"hex_code": ["8a1e3c2f4a7ffff"], "embedding": [[0.1, 0.2]], "type": ["node"]})
    ]
    mock_blob.download_to_filename.return_value = None
    sample_unified_df.to_parquet("/tmp/unified_features.parquet")

    with patch("steps.train_rl_policy.MatatuEnv", return_value=mock_matatu_env):
        with patch("stable_baselines3.common.vec_env.SubprocVecEnv") as mock_vec_env:
            mock_vec_env_instance = MagicMock()
            mock_vec_env.return_value = mock_vec_env_instance
            with patch("stable_baselines3.PPO") as mock_ppo:
                mock_ppo_instance = MagicMock()
                mock_ppo.return_value = mock_ppo_instance
                with patch("torch.nn.MSELoss", return_value=MagicMock(item=lambda: 0.1)):
                    train_hierarchical_rl_policy(sample_unified_df, n_envs=4)

    # Check logging of advanced metrics
    mock_logger.log_text.assert_any_call(pytest.approx("Phase 0 Loss: policy=0.1, aux=0.4, mean_forward_error=0.01", rel=1e-2))

def test_train_hierarchical_rl_policy_gcs_error(sample_unified_df, mock_logging_client, mock_gcs_client, mock_bigquery_client, mock_matatu_env):
    mock_logger = mock_logging_client
    mock_client, mock_blob = mock_gcs_client
    mock_read_gbq = mock_bigquery_client

    mock_read_gbq.side_effect = [
        pd.DataFrame({"hex_code": ["8a1e3c2f4a7ffff"], "road_id": [1], "edge_id": [1], "passenger_demand": [10.0]}),
        pd.DataFrame({"hex_code": ["8a1e3c2f4a7ffff"], "embedding": [[0.1, 0.2]], "type": ["node"]})
    ]
    mock_blob.download_to_filename.side_effect = Exception("GCS error")

    with patch("steps.train_rl_policy.MatatuEnv", return_value=mock_matatu_env):
        with patch("stable_baselines3.common.vec_env.SubprocVecEnv") as mock_vec_env:
            with patch("stable_baselines3.PPO"):
                with pytest.raises(Exception, match="Data load error"):
                    train_hierarchical_rl_policy(sample_unified_df, n_envs=4)

    mock_logger.log_text.assert_called_with("Data load error: GCS error")
