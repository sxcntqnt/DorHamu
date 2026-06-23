import numpy as np
import gym
from collections import deque
from typing import Dict, Any, Tuple, Optional

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer for hierarchical RL with dictionary-based obs/actions."""
    
    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Dict,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-5
    ):
        """
        Initialize the Prioritized Replay Buffer.
        
        Args:
            buffer_size: Maximum number of transitions to store
            observation_space: Dict space of observations
            action_space: Dict space of actions
            alpha: Prioritization exponent (0 = uniform sampling)
            beta: Importance sampling correction exponent
            epsilon: Small constant to ensure non-zero priorities
        """
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.pos = 0
        self.full = False
        
        # Initialize storage
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        self.max_priority = 1.0
        
        # Storage for observations (dictionary-based)
        self.observations = {
            key: np.zeros((buffer_size,) + space.shape, dtype=space.dtype)
            for key, space in observation_space.spaces.items()
        }
        self.next_observations = {
            key: np.zeros((buffer_size,) + space.shape, dtype=space.dtype)
            for key, space in observation_space.spaces.items()
        }
        
        # Storage for actions (dictionary-based)
        self.actions = {
            key: np.zeros((buffer_size,) + space.shape, dtype=space.dtype)
            for key, space in action_space.spaces.items()
        }
        
        # Storage for scalar values
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        self.infos = [None] * buffer_size
        
    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: Dict[str, np.ndarray],
        reward: float,
        next_obs: Dict[str, np.ndarray],
        done: bool,
        infos: Dict[str, Any]
    ) -> None:
        """Add a new transition to the buffer with maximum priority."""
        idx = self.pos
        
        # Store dictionary-based observations
        for key in self.observations:
            self.observations[key][idx] = obs[key]
            self.next_observations[key][idx] = next_obs[key]
        
        # Store dictionary-based actions
        for key in self.actions:
            self.actions[key][idx] = action[key]
        
        # Store scalar values
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.infos[idx] = infos
        
        # Set maximum priority for new transitions
        self.priorities[idx] = self.max_priority
        
        # Update position and buffer state
        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True
    
    def sample(
        self,
        batch_size: int,
        beta: Optional[float] = None
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions based on priorities.
        
        Args:
            batch_size: Number of transitions to sample
            beta: Importance sampling correction exponent (overrides self.beta if provided)
        
        Returns:
            samples: Dictionary containing batched transitions
            weights: Importance sampling weights
            indices: Sampled indices
        """
        beta = beta if beta is not None else self.beta
        current_size = self.buffer_size if self.full else self.pos
        
        # Compute sampling probabilities
        probs = self.priorities[:current_size] ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices
        indices = np.random.choice(current_size, batch_size, p=probs, replace=True)
        
        # Compute importance sampling weights
        weights = (current_size * probs[indices]) ** (-beta)
        weights = weights / weights.max()  # Normalize for stability
        
        # Create batch
        samples = {
            "observations": {key: self.observations[key][indices] for key in self.observations},
            "actions": {key: self.actions[key][indices] for key in self.actions},
            "rewards": self.rewards[indices],
            "next_observations": {key: self.next_observations[key][indices] for key in self.next_observations},
            "dones": self.dones[indices],
            "infos": [self.infos[idx] for idx in indices]
        }
        
        return samples, weights, indices
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Update priorities for sampled transitions.
        
        Args:
            indices: Indices of sampled transitions
            td_errors: Temporal difference errors
        """
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return self.buffer_size if self.full else self.pos
