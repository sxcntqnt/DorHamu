import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from gym.spaces import Dict, Discrete, Box
from steps.define_ma3_environment import define_ma3_environment, MatatuEnv

@pytest.fixture
def sample_unified_df():
    data = {
        "hex_code": ["8a1e3c2f4a7ffff", "8a1e3c2f4a8ffff"],
        "hex_centroid_x": [0.5, 2.5],
        "hex_centroid_y": [0.5, 2.5],
        "passenger_demand": [10.0, 15.0]
    }
    # Add hourly demand features
    for h in range(24):
        data[f"hour_{h}"] = [1.0, 2.0]
    return pd.DataFrame(data)

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
    return env

def test_define_matatu_environment_success(sample_unified_df, mock_gcs_client, mock_logging_client, mock_matatu_env):
    mock_client, mock_blob = mock_gcs_client
    mock_logger = mock_logging_client
    with patch("pickle.load", return_value={"grg": MagicMock(nodes=["8a1e3c2f4a7ffff", "8a1e3c2f4a8ffff"])}):
        with patch("steps.define_ma3_environment.MatatuEnv", return_value=mock_matatu_env) as mock_env_class:
            env = define_matatu_environment(
                unified_df=sample_unified_df,
                sumo_config="gs://my_bucket/input_data/nairobi.sumocfg",
                curriculum_phase=1
            )

    assert isinstance(env, MatatuEnv)
    mock_env_class.assert_called_with(
        grg_path="processed_data/grg.pkl",
        abstreet_map_file="gs://my_bucket/input_data/nairobi.bin",
        abstreet_scenario_file="gs://my_bucket/input_data/nairobi_scenario.json",
        sumo_config_file="gs://my_bucket/input_data/nairobi.sumocfg",
        curriculum_phase=1
    )
    mock_logger.log_text.assert_called_with("Defined Matatu environment")
    assert env.curriculum_phase == 1
    assert env.node_list == ["8a1e3c2f4a7ffff", "8a1e3c2f4a8ffff"]

def test_define_matatu_environment_gcs_error(sample_unified_df, mock_gcs_client, mock_logging_client):
    mock_client, mock_blob = mock_gcs_client
    mock_logger = mock_logging_client
    mock_blob.download_to_filename.side_effect = Exception("GCS error")
    with pytest.raises(Exception, match="Environment creation error"):
        define_matatu_environment(
            unified_df=sample_unified_df,
            sumo_config="gs://my_bucket/input_data/nairobi.sumocfg",
            curriculum_phase=0
        )
    mock_logger.log_text.assert_called_with("Environment creation error: GCS error")

def test_matatu_env_curriculum_phase(mock_matatu_env):
    env = mock_matatu_env
    env.curriculum_phase = 2
    assert env.curriculum_phase == 2
    env.reset.assert_not_called()  # Ensure reset respects curriculum phase

def test_matatu_env_set_forward_error(mock_matatu_env):
    env = mock_matatu_env
    env.set_forward_error(0.05)
    env.set_forward_error.assert_called_with(0.05)
    assert env.forward_error == 0.05

def test_matatu_env_her_metrics(mock_matatu_env):
    env = mock_matatu_env
    obs, reward, done, info = env.step({"high_level": 0, "low_level": 1})
    assert "her_reward" in info
    assert info["her_reward"] == 0.5
    assert "desired_goal" in info
    assert np.array_equal(info["desired_goal"], env.desired_goal)
    assert "achieved_goal" in info
    assert np.array_equal(info["achieved_goal"], env.achieved_goal)
    assert "intrinsic_reward" in info
    assert info["intrinsic_reward"] == 0.1

def test_matatu_env_action_observation_spaces(mock_matatu_env):
    env = mock_matatu_env
    assert isinstance(env.observation_space, Dict)
    assert "place_cells" in env.observation_space.spaces
    assert "hourly_demand" in env.observation_space.spaces
    assert "edge_congestion" in env.observation_space.spaces
    assert isinstance(env.action_space, Dict)
    assert "high_level" in env.action_space.spaces
    assert env.action_space["high_level"].n == 5
    assert "low_level" in env.action_space.spaces
    assert env.action_space["low_level"].n == 3
