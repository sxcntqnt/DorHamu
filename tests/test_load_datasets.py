import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from steps.train_hierarchical_rl_policy import train_hierarchical_rl_policy
from stable_baselines3 import PPO

@pytest.fixture
def sample_unified_df():
    return pd.DataFrame({
        "hex_code": ["8a1e3c2f4a7ffff", "8a1e3c2f4a8ffff"],
        "hex_centroid_x": [0.5, 2.5],
        "hex_centroid_y": [0.5, 2.5]
    })

@pytest.fixture
def mock_logging_client():
    with patch("google.cloud.logging.Client") as mock_client:
        mock_logger = MagicMock()
        mock_client.return_value.logger.return_value = mock_logger
        yield mock_logger

def test_train_hierarchical_rl_policy_success(sample_unified_df, mock_logging_client):
    mock_logger = mock_logging_client
    with patch("stable_baselines3.PPO") as mock_ppo:
        mock_ppo_instance = MagicMock()
        mock_ppo.return_value = mock_ppo_instance
        with patch("stable_baselines3.common.vec_env.DummyVecEnv") as mock_vec_env:
            mock_vec_env_instance = MagicMock()
            mock_vec_env.return_value = mock_vec_env_instance
            high_level_policy, low_level_policy = train_hierarchical_rl_policy(sample_unified_df)
    
    assert isinstance(high_level_policy, PPO)
    assert isinstance(low_level_policy, PPO)
    mock_ppo.assert_called()
    mock_logger.log_text.assert_called_with("Trained hierarchical RL policies")

def test_train_hierarchical_rl_policy_empty_input(mock_logging_client):
    mock_logger = mock_logging_client
    empty_df = pd.DataFrame()
    with patch("stable_baselines3.PPO") as mock_ppo:
        with patch("stable_baselines3.common.vec_env.DummyVecEnv") as mock_vec_env:
            high_level_policy, low_level_policy = train_hierarchical_rl_policy(empty_df)
    
    assert isinstance(high_level_policy, PPO)
    assert isinstance(low_level_policy, PPO)
    mock_logger.log_text.assert_called_with("Trained hierarchical RL policies")
