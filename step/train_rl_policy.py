from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import HerReplayBuffer
from utils.PER import  PrioritizedReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
import torch.nn as nn
import numpy as np
from google.cloud import logging
from google.cloud import storage
import pandas as pd
import bigframes.pandas as bpd
import os
from zenml import step
from typing import Tuple
from .define_ma3_environment import MatatuEnv

@step
def train_hierarchical_rl_policy(unified_df: pd.DataFrame, n_envs: int = 4) -> Tuple[PPO, PPO]:
    """Train high-level and low-level PPO policies for Matatu route finding, balancing client and driver needs."""
    # Initialize logging
    logger = logging.Client().logger("matatu_pipeline")

    # Load data
    try:
        gcs_client = storage.Client(project="my_project")
        bucket = gcs_client.bucket("my_bucket")
        blob = bucket.blob("processed_data/unified_features.parquet")
        blob.download_to_filename("/tmp/unified_features.parquet")
        unified_df = pd.read_parquet("/tmp/unified_features.parquet")

        h3_mappings = bpd.read_gbq("SELECT hex_code, road_id, edge_id, passenger_demand FROM my_project.my_dataset.h3_mappings").to_pandas()
        grg_embeddings_df = bpd.read_gbq("SELECT hex_code, embedding, type FROM my_project.my_dataset.grg_embeddings").to_pandas()
        logger.log_text("Loaded unified features, H3 mappings, and GRG embeddings")
    except Exception as e:
        logger.log_text(f"Data load error: {str(e)}")
        raise

    # Forward model for curiosity-driven exploration
    class ForwardModel(nn.Module):
        def __init__(self, obs_dim, action_dim, hidden_dim=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, obs_dim)
            )
        def forward(self, obs, action):
            return self.net(torch.cat([obs, action], dim=-1))

    # Custom policy with auxiliary heads
    class DictPolicy(ActorCriticPolicy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.net_arch = [dict(pi=[512, 256], vf=[512, 256])]
            # Auxiliary heads
            self.aux_demand_head = nn.Linear(256, 24)  # Predict hourly_demand
            self.aux_congestion_head = nn.Linear(256, 1)  # Predict edge_congestion
            self.reward_head = nn.Linear(256, 1)  # Predict reward
            obs_space = kwargs["observation_space"]
            self.forward_model = ForwardModel(
                obs_dim=obs_space["place_cells"].shape[0],
                action_dim=obs_space["high_level"].n + obs_space["low_level"].n
            )

        def forward(self, obs, action=None, *args, **kwargs):
            features = self.extract_features(obs)
            value, actor_features = self.forward_actor_critic(features)
            # Auxiliary predictions
            demand_pred = self.aux_demand_head(actor_features)
            congestion_pred = self.aux_congestion_head(actor_features)
            reward_pred = self.reward_head(actor_features)
            # Forward model prediction
            action_tensor = torch.zeros(obs["high_level"].shape[0], self.action_space["high_level"].n + self.action_space["low_level"].n, device=obs["high_level"].device)
            if action is not None:
                high_level_idx = action["high_level"] if isinstance(action["high_level"], torch.Tensor) else torch.tensor(action["high_level"], device=obs["high_level"].device)
                low_level_idx = action["low_level"] if isinstance(action["low_level"], torch.Tensor) else torch.tensor(action["low_level"], device=obs["high_level"].device)
                action_tensor.scatter_(1, high_level_idx.unsqueeze(1), 1)
                action_tensor.scatter_(1, (low_level_idx + self.action_space["high_level"].n).unsqueeze(1), 1)
            forward_pred = self.forward_model(obs["place_cells"], action_tensor)
            return value, actor_features, {
                "demand_pred": demand_pred,
                "congestion_pred": congestion_pred,
                "reward_pred": reward_pred,
                "forward_pred": forward_pred
            }

    # Create environment with curriculum
    def make_env(curriculum_phase=0):
        try:
            env = MatatuEnv(
                grg_path="processed_data/grg.pkl",
                abstreet_map_file="gs://my_bucket/input_data/nairobi.bin",
                abstreet_scenario_file="gs://my_bucket/input_data/nairobi_scenario.json",
                sumo_config_file="gs://my_bucket/input_data/nairobi.sumocfg",
                curriculum_phase=curriculum_phase
            )
            logger.log_text(f"Created MatatuEnv with curriculum phase {curriculum_phase}")
            return env
        except Exception as e:
            logger.log_text(f"Environment creation error: {str(e)}")
            raise

    # Training function with HER, PER, and auxiliary losses
    def train_policy(policy, env, eval_env, curriculum_phases, timesteps_per_phase, n_envs=4, is_high_level=True):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            policy.to(device)
            for phase in curriculum_phases:
                logger.log_text(f"{'High-level' if is_high_level else 'Low-level'} policy: Starting curriculum phase {phase}")
                env = SubprocVecEnv([lambda: make_env(curriculum_phase=phase) for _ in range(n_envs)])
                eval_env = SubprocVecEnv([lambda: make_env(curriculum_phase=phase) for _ in range(n_envs)])

                # Initialize replay buffers
                her_buffer = HindsightReplayBuffer(
                    buffer_size=100000,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    goal_selection_strategy="future",
                    n_sampled_goal=4
                )
                per_buffer = PrioritizedReplayBuffer(
                    buffer_size=100000,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    alpha=0.6
                )

                eval_callback = EvalCallback(
                    eval_env,
                    best_model_save_path=f"/tmp/{'high' if is_high_level else 'low'}_level_best_model",
                    log_path=f"./tensorboard/{'high' if is_high_level else 'low'}_level/",
                    eval_freq=5000,
                    deterministic=True,
                    render=False,
                    callback_on_new_best=StopTrainingOnRewardThreshold(
                        reward_threshold=50.0 if is_high_level else 30.0,
                        verbose=1
                    )
                )

                policy.set_env(env)
                total_timesteps = timesteps_per_phase if is_high_level else timesteps_per_phase // 2
                n_steps = policy.n_steps // n_envs  # Adjust for parallel environments

                for _ in range(total_timesteps // n_steps):
                    # Collect rollouts
                    obs = env.reset()
                    for _ in range(n_steps):
                        action, _ = policy.predict(obs, deterministic=False)
                        next_obs, rewards, dones, infos = env.step(action)

                        # Store in buffers
                        forward_errors = []
                        for i in range(n_envs):
                            her_buffer.add(
                                obs=obs[i],
                                next_obs=next_obs[i],
                                action=action[i],
                                reward=infos[i]["her_reward"],
                                done=dones[i],
                                infos=[infos[i]],
                                desired_goal=infos[i]["desired_goal"],
                                achieved_goal=infos[i]["achieved_goal"]
                            )
                            per_buffer.add(
                                obs=obs[i],
                                next_obs=next_obs[i],
                                action=action[i],
                                reward=rewards[i],
                                done=dones[i],
                                infos=[infos[i]]
                            )

                            # Update forward error for each environment
                            obs_tensor = {k: torch.tensor(obs[i][k], device=device).unsqueeze(0) for k in obs[i]}
                            _, _, aux_preds = policy.forward(obs_tensor, action[i])
                            forward_pred = aux_preds["forward_pred"]
                            forward_error = nn.MSELoss()(forward_pred, torch.tensor(next_obs[i]["place_cells"], device=device)).item()
                            forward_errors.append(forward_error)
                            env.env_method("set_forward_error", forward_error, indices=[i])

                        obs = next_obs
                        if any(dones):
                            obs = env.reset()

                    # Train with PPO, HER, and auxiliary losses
                    policy.train()
                    batch = her_buffer.sample(policy.batch_size, env=env)
                    per_batch = per_buffer.sample(policy.batch_size, beta=0.4)

                    # Convert batch to tensors
                    batch_obs = {k: torch.tensor(v, device=device) for k, v in batch.observations.items()}
                    batch_actions = {
                        "high_level": torch.tensor([a["high_level"] for a in batch.actions], device=device),
                        "low_level": torch.tensor([a["low_level"] for a in batch.actions], device=device)
                    }
                    batch_next_obs = {k: torch.tensor(v, device=device) for k, v in batch.next_observations.items()}
                    batch_rewards = torch.tensor(batch.rewards, device=device)
                    batch_dones = torch.tensor(batch.dones, device=device)

                    # Compute auxiliary losses
                    _, _, aux_preds = policy.forward(batch_obs, batch_actions)
                    demand_loss = nn.MSELoss()(aux_preds["demand_pred"], batch_obs["hourly_demand"])
                    congestion_loss = nn.MSELoss()(aux_preds["congestion_pred"], batch_obs["edge_congestion"])
                    reward_loss = nn.MSELoss()(aux_preds["reward_pred"], batch_rewards)
                    forward_loss = nn.MSELoss()(aux_preds["forward_pred"], batch_next_obs["place_cells"])
                    aux_loss = 0.1 * (demand_loss + congestion_loss + reward_loss + forward_loss)

                    # Update policy
                    policy_loss = policy.compute_loss(batch)
                    total_loss = policy_loss + aux_loss
                    policy.optimizer.zero_grad()
                    total_loss.backward()
                    policy.optimizer.step()

                    # Update PER priorities
                    td_errors = policy.compute_td_error(per_batch)
                    per_buffer.update_priorities(per_batch.indices, td_errors.cpu().numpy())

                    # Log losses
                    logger.log_text(f"Phase {phase} Loss: policy={policy_loss.item()}, aux={aux_loss.item()}, mean_forward_error={np.mean(forward_errors)}")

                policy.learn(total_timesteps=1000, callback=eval_callback, progress_bar=True)  # Final PPO update

            policy.save(f"/tmp/matatu_{'high' if is_high_level else 'low'}_level_policy.zip")
            bucket.blob(f"models/matatu_{'high' if is_high_level else 'low'}_level_policy.zip").upload_from_filename(
                f"/tmp/matatu_{'high' if is_high_level else 'low'}_level_policy.zip"
            )
            logger.log_text(f"Trained and saved {'high-level' if is_high_level else 'low-level'} policy to GCS")
        except Exception as e:
            logger.log_text(f"{'High-level' if is_high_level else 'Low-level'} policy training error: {str(e)}")
            raise
        finally:
            env.close()
            eval_env.close()

    # Train high-level policy
    try:
        env = SubprocVecEnv([lambda: make_env(curriculum_phase=0) for _ in range(n_envs)])
        eval_env = SubprocVecEnv([lambda: make_env(curriculum_phase=0) for _ in range(n_envs)])

        high_level_policy = PPO(
            DictPolicy,
            env,
            verbose=1,
            learning_rate=lambda f: 3e-4 * (1 - f),
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            tensorboard_log="./tensorboard/high_level/",
            policy_kwargs={"net_arch": [dict(pi=[512, 256], vf=[512, 256])]}
        )

        curriculum_phases = [0, 1, 2]
        timesteps_per_phase = 150000 // len(curriculum_phases)  # Reduced due to sample efficiency
        train_policy(high_level_policy, env, eval_env, curriculum_phases, timesteps_per_phase, n_envs=n_envs, is_high_level=True)
    except Exception as e:
        logger.log_text(f"High-level policy training error: {str(e)}")
        raise

    # Train low-level policy
    try:
        env = SubprocVecEnv([lambda: make_env(curriculum_phase=0) for _ in range(n_envs)])
        eval_env = SubprocVecEnv([lambda: make_env(curriculum_phase=0) for _ in range(n_envs)])

        low_level_policy = PPO(
            DictPolicy,
            env,
            verbose=1,
            learning_rate=lambda f: 5e-4 * (1 - f),
            n_steps=1024,
            batch_size=32,
            n_epochs=5,
            gamma=0.95,
            gae_lambda=0.9,
            clip_range=0.1,
            tensorboard_log="./tensorboard/low_level/",
            policy_kwargs={"net_arch": [dict(pi=[256, 128], vf=[256, 128])]}
        )

        train_policy(low_level_policy, env, eval_env, curriculum_phases, timesteps_per_phase, n_envs=n_envs, is_high_level=False)
    except Exception as e:
        logger.log_text(f"Low-level policy training error: {str(e)}")
        raise

    return high_level_policy, low_level_policy
