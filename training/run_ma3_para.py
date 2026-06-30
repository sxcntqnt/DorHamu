import multiprocessing as mp
import numpy as np
import ctypes
import math
import requests
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass

from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.nn.optim import Adam
import tinygrad.nn as nn

from utils.utils import Counter


# ---------------------------------------------------------------------------
# Tinygrad NN Model
# Replaces NN_Model from models/ppo_model.py
# Actor-Critic with shared trunk, separate policy head (mean+log_std) and value head
# ---------------------------------------------------------------------------

class MatatuNN:
    def __init__(self, state_size: int, action_size: int, hidden_size: int):
        self.trunk = [
            Linear(state_size, hidden_size),
            Linear(hidden_size, hidden_size),
        ]
        # Policy head outputs mean and log_std for Gaussian action distribution
        self.policy_mean = Linear(hidden_size, action_size)
        self.policy_log_std = Linear(hidden_size, action_size)
        # Value head
        self.value_head = Linear(hidden_size, 1)

        self.action_size = action_size
        self.hidden_size = hidden_size

    def parameters(self) -> List[Tensor]:
        params = []
        for layer in self.trunk:
            params += [layer.weight, layer.bias]
        for layer in [self.policy_mean, self.policy_log_std, self.value_head]:
            params += [layer.weight, layer.bias]
        return params

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns (mean, std, value).
        mean/std: shape (batch, action_size)
        value: shape (batch, 1)
        """
        h = x
        for layer in self.trunk:
            h = layer(h).relu()

        mean = self.policy_mean(h)
        # Clamp log_std to prevent std collapsing to zero or exploding
        log_std = self.policy_log_std(h).clip(-3, 1)
        std = log_std.exp()
        value = self.value_head(h)
        return mean, std, value

    def get_flat_params(self) -> np.ndarray:
        """Flatten all parameters to a 1D numpy array for shared memory sync."""
        return np.concatenate([
            p.numpy().flatten() for p in self.parameters()
        ])

    def set_flat_params(self, flat: np.ndarray):
        """Restore parameters from a flat numpy array."""
        offset = 0
        for p in self.parameters():
            size = int(np.prod(p.shape))
            p.assign(
                Tensor(flat[offset:offset + size].reshape(p.shape))
            )
            offset += size

    def param_count(self) -> int:
        return sum(int(np.prod(p.shape)) for p in self.parameters())


# ---------------------------------------------------------------------------
# Shared weight buffer — replaces PyTorch share_memory()
# Uses mp.Array (raw C float array in shared memory) + mp.Lock
# ---------------------------------------------------------------------------

class SharedWeightBuffer:
    """
    Holds network weights in a shared memory float array.
    All processes read/write through this with explicit locking.

    This replaces PyTorch's share_memory() + Hogwild gradient push.
    We use a coarser but cleaner pattern: full weight sync at rollout
    boundaries rather than async per-parameter gradient injection,
    which is safer given each episode takes seconds not milliseconds.
    """

    def __init__(self, param_count: int):
        self.lock = mp.Lock()
        # mp.Array('f', n) allocates n floats in shared memory
        self.buffer = mp.Array(ctypes.c_float, param_count)
        self.param_count = param_count

    def write(self, flat_params: np.ndarray):
        with self.lock:
            self.buffer[:] = flat_params.astype(np.float32).tolist()

    def read(self) -> np.ndarray:
        with self.lock:
            return np.array(self.buffer[:], dtype=np.float32)

    def apply_grad_update(self, flat_grads: np.ndarray, lr: float):
        """
        Applies a gradient update directly to shared weights (SGD-style).
        Called by training workers instead of a shared optimizer.
        Lock held for the full read-modify-write cycle.
        """
        with self.lock:
            current = np.array(self.buffer[:], dtype=np.float32)
            updated = current - lr * flat_grads
            self.buffer[:] = updated.tolist()


# ---------------------------------------------------------------------------
# Gaussian policy helpers — replaces torch.distributions.Normal
# ---------------------------------------------------------------------------

def gaussian_sample(mean: Tensor, std: Tensor) -> Tensor:
    """Reparameterized sample: mean + std * N(0,1)"""
    noise = Tensor.randn(*mean.shape)
    return mean + std * noise


def gaussian_log_prob(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """
    Log probability of x under N(mean, std²).
    Returns shape (batch, action_size) — sum across action dims for scalar log_prob.
    """
    var = std ** 2
    log_std = std.log()
    return (
        -0.5 * ((x - mean) ** 2 / (var + 1e-8))
        - log_std
        - Tensor([math.log(math.sqrt(2 * math.pi))])
    )


def gaussian_entropy(std: Tensor) -> Tensor:
    """
    Entropy of N(mean, std²): 0.5 * log(2πe * std²)
    Returns shape (batch, action_size).
    """
    return 0.5 * (1 + (2 * math.pi * math.e * std ** 2 + 1e-8).log())


def clip_grad_norm(params: List[Tensor], max_norm: float) -> float:
    """
    Computes global gradient norm and clips in place.
    Replaces torch.nn.utils.clip_grad_norm_.
    Returns the pre-clip global norm for logging.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0

    global_norm = float(np.sqrt(sum(
        np.sum(g.numpy() ** 2) for g in grads
    )))

    if global_norm > max_norm:
        scale = max_norm / (global_norm + 1e-8)
        for p in params:
            if p.grad is not None:
                p.grad = Tensor(p.grad.numpy() * scale)

    return global_norm


# ---------------------------------------------------------------------------
# Per-episode transition (unchanged from PyTorch version)
# ---------------------------------------------------------------------------

@dataclass
class EpisodeTransition:
    action: np.ndarray
    log_prob: float
    reward: float
    value: float
    episode_id: int
    stats: dict


# ---------------------------------------------------------------------------
# Matatu PPO Worker — tinygrad version
# ---------------------------------------------------------------------------

class MatatuPPOWorker:

    def __init__(
        self,
        worker_id: str,
        shared_buffer: SharedWeightBuffer,
        local_NN: MatatuNN,
        episode_runner,
        action_space,
        constants: dict,
    ):
        self.worker_id = worker_id
        self.shared_buffer = shared_buffer
        self.local_NN = local_NN
        self.runner = episode_runner
        self.action_space = action_space
        self.constants = constants
        self._episode_count = 0

        # Local Adam optimizer — each worker owns its own optimizer state
        # Global weight sync happens via shared_buffer, not shared optimizer
        self.optimizer = Adam(
            self.local_NN.parameters(),
            lr=constants['ppo']['learning_rate']
        )

    def _sync_local_from_shared(self):
        flat = self.shared_buffer.read()
        self.local_NN.set_flat_params(flat)

    def _push_local_to_shared(self):
        flat = self.local_NN.get_flat_params()
        self.shared_buffer.write(flat)

    def _sample_action(self) -> Tuple[np.ndarray, float, float]:
        # Unconditional policy: zero context vector input
        dummy = Tensor.zeros(1, self.constants['ppo']['hidden_layer_size'])
        Tensor.no_grad = True
        mean, std, value = self.local_NN.forward(dummy)
        Tensor.no_grad = False

        action = gaussian_sample(mean, std)
        log_prob = gaussian_log_prob(action, mean, std).sum().numpy().item()
        value_est = value.numpy().item()

        return action.numpy().squeeze(), log_prob, value_est

    def train_rollout(self, rollout_step: int):
        self._sync_local_from_shared()

        transitions: List[EpisodeTransition] = []
        episodes_per_rollout = self.constants['episode']['episodes_per_rollout']

        for _ in range(episodes_per_rollout):
            raw_action, log_prob, value = self._sample_action()
            reward, stats = self.runner.run_episode(
                raw_ppo_action=raw_action,
                episode_id=self._episode_count,
            )
            self._episode_count += 1
            transitions.append(EpisodeTransition(
                action=raw_action,
                log_prob=log_prob,
                reward=reward,
                value=value,
                episode_id=self._episode_count,
                stats=stats,
            ))

        self._ppo_update(transitions)
        # Push updated weights back to shared buffer after each rollout
        self._push_local_to_shared()

    def _ppo_update(self, transitions: List[EpisodeTransition]):
        clip_eps = self.constants['ppo']['clip_epsilon']
        vf_coef = self.constants['ppo']['value_loss_coef']
        ent_coef = self.constants['ppo']['entropy_coef']
        max_grad_norm = self.constants['ppo']['max_grad_norm']

        actions_np = np.stack([t.action for t in transitions])
        old_log_probs_np = np.array([t.log_prob for t in transitions], dtype=np.float32)
        rewards_np = np.array([t.reward for t in transitions], dtype=np.float32)
        old_values_np = np.array([t.value for t in transitions], dtype=np.float32)

        actions = Tensor(actions_np)
        old_log_probs = Tensor(old_log_probs_np)
        rewards = Tensor(rewards_np)
        old_values = Tensor(old_values_np)

        # Normalize returns
        rewards_mean = float(rewards_np.mean())
        rewards_std = float(rewards_np.std()) + 1e-8
        returns = (rewards - rewards_mean) / rewards_std

        advantages = returns - old_values
        adv_mean = float(advantages.numpy().mean())
        adv_std = float(advantages.numpy().std()) + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # Forward pass under current policy
        dummy = Tensor.zeros(
            len(transitions),
            self.constants['ppo']['hidden_layer_size']
        )
        mean, std, values_pred = self.local_NN.forward(dummy)

        new_log_probs = gaussian_log_prob(actions, mean, std).sum(axis=-1)
        entropy = gaussian_entropy(std).sum(axis=-1).mean()

        # PPO clipped surrogate
        log_ratio = new_log_probs - old_log_probs
        ratio = log_ratio.exp()
        surr1 = ratio * advantages
        surr2 = ratio.clip(1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -surr1.minimum(surr2).mean()

        # Value loss
        value_loss = (values_pred.squeeze() - returns).pow(2).mean()

        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm(self.local_NN.parameters(), max_grad_norm)
        self.optimizer.step()


# ---------------------------------------------------------------------------
# Process target functions
# ---------------------------------------------------------------------------

def matatu_train_worker(
    worker_id: str,
    shared_buffer: SharedWeightBuffer,
    rollout_counter: Counter,
    constants: dict,
    routes,
    map_json_path: str,
    map_bin_path: str,
    scenario_bin_dir: str,
):
    from matatu.scenario_generator import (
        MatatuScenarioGenerator, MatatuActionSpace,
        ABStreetMapIndex, NairobiRushHourWindows
    )
    from matatu.episode_runner import MatatuEpisodeRunner

    port = 1234 + int(worker_id.split('_')[-1])
    host = f"http://localhost:{port}"

    map_index = ABStreetMapIndex(map_json_path)
    action_space = MatatuActionSpace(routes)

    generator = MatatuScenarioGenerator(
        routes=routes,
        map_index=map_index,
        action_space=action_space,
        rush_hours=NairobiRushHourWindows(),
        background_vehicles=constants['scenario']['background_vehicles'],
        rng_seed=int(worker_id.split('_')[-1]),
    )

    runner = MatatuEpisodeRunner(
        generator=generator,
        map_bin_path=map_bin_path,
        scenario_bin_dir=scenario_bin_dir,
        host=host,
        simulate_until=constants['scenario']['simulate_until'],
    )

    action_dims = action_space.total_dims
    local_NN = MatatuNN(
        state_size=constants['ppo']['hidden_layer_size'],
        action_size=action_dims,
        hidden_size=constants['ppo']['hidden_layer_size'],
    )
    # Initialize from shared buffer so all workers start from same weights
    local_NN.set_flat_params(shared_buffer.read())

    worker = MatatuPPOWorker(
        worker_id=worker_id,
        shared_buffer=shared_buffer,
        local_NN=local_NN,
        episode_runner=runner,
        action_space=action_space,
        constants=constants,
    )

    while rollout_counter.get() < constants['episode']['num_train_rollouts'] + 1:
        worker.train_rollout(rollout_counter.get())
        rollout_counter.increment()


def matatu_eval_worker(
    worker_id: str,
    shared_buffer: SharedWeightBuffer,
    rollout_counter: Counter,
    data_collector,
    constants: dict,
    routes,
    map_json_path: str,
    map_bin_path: str,
    scenario_bin_dir: str,
):
    from matatu.scenario_generator import (
        MatatuScenarioGenerator, MatatuActionSpace,
        ABStreetMapIndex, NairobiRushHourWindows
    )
    from matatu.episode_runner import MatatuEpisodeRunner

    n_train_workers = constants['parallel']['num_workers']
    port = 1234 + n_train_workers + int(worker_id.split('_')[-1])
    host = f"http://localhost:{port}"

    map_index = ABStreetMapIndex(map_json_path)
    action_space = MatatuActionSpace(routes)

    generator = MatatuScenarioGenerator(
        routes=routes,
        map_index=map_index,
        action_space=action_space,
        rush_hours=NairobiRushHourWindows(),
        background_vehicles=constants['scenario']['background_vehicles'],
        rng_seed=9999,
    )

    runner = MatatuEpisodeRunner(
        generator=generator,
        map_bin_path=map_bin_path,
        scenario_bin_dir=scenario_bin_dir,
        host=host,
        simulate_until=constants['scenario']['simulate_until'],
    )

    action_dims = action_space.total_dims
    local_NN = MatatuNN(
        state_size=constants['ppo']['hidden_layer_size'],
        action_size=action_dims,
        hidden_size=constants['ppo']['hidden_layer_size'],
    )

    last_eval = 0
    episode_id = 0

    while True:
        curr_r = rollout_counter.get()

        if curr_r % constants['episode']['eval_freq'] == 0 and last_eval != curr_r:
            last_eval = curr_r
            local_NN.set_flat_params(shared_buffer.read())

            eval_rewards = []
            for _ in range(constants['episode']['eval_num_eps']):
                # Deterministic eval: use mean action, no sampling noise
                dummy = Tensor.zeros(1, constants['ppo']['hidden_layer_size'])
                Tensor.no_grad = True
                mean, _, _ = local_NN.forward(dummy)
                Tensor.no_grad = False
                raw_action = mean.numpy().squeeze()

                reward, stats = runner.run_episode(
                    raw_ppo_action=raw_action,
                    episode_id=episode_id,
                )
                episode_id += 1
                eval_rewards.append(reward)

                if data_collector:
                    data_collector.log(rollout=curr_r, **stats)

            print(f"[Eval @ rollout {curr_r}] mean_reward={np.mean(eval_rewards):.4f}")

        if curr_r >= constants['episode']['num_train_rollouts'] + 1:
            break


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train_matatu_PPO(
    constants: dict,
    data_collector,
    routes,
    map_json_path: str,
    map_bin_path: str,
    scenario_bin_dir: str,
):
    from matatu.scenario_generator import MatatuActionSpace

    action_dims = MatatuActionSpace(routes).total_dims
    hidden_size = constants['ppo']['hidden_layer_size']

    # Initialize one network to get param count, then seed shared buffer
    seed_NN = MatatuNN(
        state_size=hidden_size,
        action_size=action_dims,
        hidden_size=hidden_size,
    )
    shared_buffer = SharedWeightBuffer(seed_NN.param_count())
    shared_buffer.write(seed_NN.get_flat_params())  # broadcast initial weights

    rollout_counter = Counter()
    processes = []

    # Eval worker
    p = mp.Process(
        target=matatu_eval_worker,
        args=(
            'eval_0', shared_buffer, rollout_counter, data_collector,
            constants, routes, map_json_path, map_bin_path, scenario_bin_dir,
        )
    )
    p.start()
    processes.append(p)

    # Training workers — no shared optimizer; each owns its own Adam state
    for i in range(constants['parallel']['num_workers']):
        p = mp.Process(
            target=matatu_train_worker,
            args=(
                f'train_{i}', shared_buffer, rollout_counter,
                constants, routes, map_json_path, map_bin_path, scenario_bin_dir,
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()