"""
Phase 4: Deep Q-Network (DQN) Agent

Neural network agent with:
- Double DQN (reduces overestimation bias)
- Vectorized numpy replay buffer (fast training)
- State normalization (stable learning)
- Action masking (prevents invalid actions)
"""

import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Q-value network: maps state -> Q-values for all actions.

    Architecture: 2 hidden layers with ReLU activation.
    Input:  normalized state vector (20-dim)
    Output: Q-values for each action slot (5-dim)
    """

    def __init__(self, state_dim=20, n_actions=5, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class NumpyReplayBuffer:
    """
    High-performance replay buffer using pre-allocated numpy arrays.

    Stores transitions with pre-computed action masks for vectorized
    Double DQN updates. Much faster than deque-based alternatives.
    """

    def __init__(self, capacity, state_dim, n_actions):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        # Pre-computed mask: 0.0 for valid next-actions, -1e9 for invalid
        self.next_masks = np.full((capacity, n_actions), -1e9, dtype=np.float32)
        self._idx = 0
        self._size = 0

    def push(self, state, action, reward, next_state, done, available_next):
        """Store a transition with pre-computed action mask."""
        i = self._idx % self.capacity
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = done
        self.next_masks[i] = -1e9
        for a in available_next:
            self.next_masks[i, a] = 0.0
        self._idx += 1
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample a random batch as numpy arrays."""
        idx = np.random.choice(self._size, batch_size, replace=False)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
            self.next_masks[idx],
        )

    def __len__(self):
        return self._size


class DQNAgent:
    """
    DQN agent with Double DQN, target network, replay buffer,
    action masking, and state normalization.
    """

    def __init__(
        self,
        state_dim=20,
        n_actions=5,
        hidden=64,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.998,
        epsilon_min=0.02,
        buffer_capacity=20_000,
        batch_size=64,
        target_update_freq=50,
        state_mean=None,
        state_std=None,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # State normalization
        self.state_mean = state_mean if state_mean is not None else np.zeros(state_dim)
        self.state_std = state_std if state_std is not None else np.ones(state_dim)

        # Networks
        self.q_net = QNetwork(state_dim, n_actions, hidden)
        self.target_net = QNetwork(state_dim, n_actions, hidden)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = NumpyReplayBuffer(buffer_capacity, state_dim, n_actions)

        self.losses = []

    def normalize(self, state):
        """Normalize state features."""
        return (state - self.state_mean) / self.state_std

    def select_action(self, state, available_actions):
        """Epsilon-greedy action selection with masking on normalized state."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(available_actions)
        s = self.normalize(state)
        with torch.no_grad():
            q_vals = self.q_net(torch.FloatTensor(s)).numpy()
        masked = np.full(self.n_actions, -np.inf)
        masked[available_actions] = q_vals[available_actions]
        return int(np.argmax(masked))

    def select_action_greedy(self, state, available_actions):
        """Pure greedy selection (for evaluation)."""
        s = self.normalize(state)
        with torch.no_grad():
            q_vals = self.q_net(torch.FloatTensor(s)).numpy()
        masked = np.full(self.n_actions, -np.inf)
        masked[available_actions] = q_vals[available_actions]
        return int(np.argmax(masked))

    def store_transition(self, state, action, reward, next_state, done, available_next):
        """Store normalized transition in buffer."""
        sn = self.normalize(state)
        nsn = self.normalize(next_state) if not done else np.zeros(self.state_dim)
        self.buffer.push(sn, action, reward, nsn, float(done), available_next)

    def train_step(self):
        """
        One gradient step using Double DQN with vectorized masking.

        Double DQN: online net selects best next-action, target net
        evaluates it. This reduces overestimation bias.
        """
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, ns, d, nmask = self.buffer.sample(self.batch_size)

        s_t = torch.FloatTensor(s)
        a_t = torch.LongTensor(a)
        r_t = torch.FloatTensor(r)
        ns_t = torch.FloatTensor(ns)
        d_t = torch.FloatTensor(d)
        nm_t = torch.FloatTensor(nmask)

        q_current = self.q_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: select with online, evaluate with target
            q_next_online = self.q_net(ns_t) + nm_t  # mask invalid
            best_actions = q_next_online.argmax(dim=1)
            q_next_val = self.target_net(ns_t).gather(
                1, best_actions.unsqueeze(1)
            ).squeeze(1)
            q_next_val = q_next_val * (1.0 - d_t)  # zero for terminal
            q_target = r_t + self.gamma * q_next_val

        loss = nn.functional.smooth_l1_loss(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.losses.append(loss.item())
        return loss.item()

    def update_target_network(self):
        """Sync target network with online network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    @staticmethod
    def compute_normalization_stats(n_episodes=300, n_tasks=5):
        """
        Collect state statistics from random episodes for normalization.

        Returns:
            state_mean, state_std: numpy arrays of shape (state_dim,)
        """
        from src.environment import TaskEnv

        states = []
        for ep in range(n_episodes):
            env = TaskEnv(n_tasks=n_tasks)
            s = env.reset()
            states.append(s)
            while not env.done:
                s, _, _, _ = env.step(np.random.choice(env.available_actions()))
                if s is not None:
                    states.append(s)
        all_states = np.array(states)
        return all_states.mean(axis=0), all_states.std(axis=0) + 1e-8
