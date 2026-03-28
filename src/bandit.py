"""
Phase 3: Contextual Bandit with ε-greedy Exploration

Linear bandit agent that learns online from interaction.
One learned model per action (task slot) using MSE loss.
"""

import numpy as np


class EpsilonGreedyBandit:
    """
    Linear contextual bandit using ε-greedy exploration.
    
    For each action (task slot), maintains a linear model:
        Q(s, a) = w_a · φ(s)
    
    where φ(s) is the state features with bias term appended.
    """
    
    def __init__(
        self,
        n_actions,
        state_dim,
        lr=0.01,
        epsilon=0.2,
        epsilon_decay=0.995,
        epsilon_min=0.05,
    ):
        """
        Initialize bandit agent.
        
        Args:
            n_actions: number of actions (typically MAX_TASKS)
            state_dim: dimension of state features (typically MAX_TASKS * 4)
            lr: learning rate for gradient updates
            epsilon: initial exploration rate
            epsilon_decay: decay factor per episode
            epsilon_min: minimum exploration rate (floor)
        """
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.feature_dim = state_dim + 1  # +1 for bias term
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # One weight vector per action
        self.weights = np.zeros((n_actions, self.feature_dim))
        
        # For diagnostics: track weight norms per action
        self.weight_norms = []
    
    def _features(self, state):
        """
        Append bias term to state vector.
        
        Args:
            state: np.array of shape (state_dim,)
        
        Returns:
            np.array of shape (state_dim + 1,) with bias at end
        """
        return np.append(state, 1.0)
    
    def predict(self, state, available_actions):
        """
        Predict Q-values for available actions.
        
        Args:
            state: np.array of shape (state_dim,)
            available_actions: list of valid action indices
        
        Returns:
            dict: {action_idx: q_value}
        """
        phi = self._features(state)
        return {a: self.weights[a] @ phi for a in available_actions}
    
    def select_action(self, state, available_actions):
        """
        Select action using ε-greedy policy.
        
        With probability ε: random exploration.
        With probability 1-ε: exploitation (argmax Q-value).
        
        Args:
            state: np.array of shape (state_dim,)
            available_actions: list of valid action indices
        
        Returns:
            int: selected action
        """
        if np.random.rand() < self.epsilon:
            # Explore: random action from available
            return np.random.choice(available_actions)
        else:
            # Exploit: greedy
            q_values = self.predict(state, available_actions)
            return max(q_values, key=q_values.get)
    
    def update(self, state, action, reward):
        """
        Online gradient update for a single transition.
        
        Uses MSE loss: L = 0.5 * (r - Q(s,a))^2
        Gradient: ∇_w L = -(r - Q(s,a)) * φ(s)
        Update: w_a := w_a + lr * (r - Q(s,a)) * φ(s)
        
        Args:
            state: np.array of shape (state_dim,)
            action: int, selected action
            reward: float, observed reward
        """
        phi = self._features(state)
        q_pred = self.weights[action] @ phi
        error = reward - q_pred
        
        # Gradient step
        self.weights[action] += self.lr * error * phi
    
    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_weight_norms(self):
        """Diagnostic: L2 norm of each action's weight vector."""
        return np.linalg.norm(self.weights, axis=1)
