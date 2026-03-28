"""
Phase 3: Train Contextual Bandit

Training loop for ε-greedy linear bandit over 2000 episodes.
Saves trained agent to models/bandit.pkl
"""

import numpy as np
import pickle
from pathlib import Path

from src.environment import TaskEnv, MAX_TASKS
from src.bandit import EpsilonGreedyBandit


def train_bandit(
    n_episodes=2000,
    n_tasks=MAX_TASKS,
    lr=0.01,
    epsilon=0.2,
    epsilon_decay=0.995,
    epsilon_min=0.05,
    verbose=True,
):
    """
    Train a contextual bandit agent.
    
    Args:
        n_episodes: number of training episodes
        n_tasks: number of tasks per episode
        lr: learning rate
        epsilon: initial exploration rate
        epsilon_decay: decay factor per episode
        epsilon_min: minimum exploration rate
        verbose: print progress
    
    Returns:
        agent: trained EpsilonGreedyBandit
        episode_rewards: array of episode rewards
    """
    state_dim = n_tasks * 4
    agent = EpsilonGreedyBandit(
        n_actions=n_tasks,
        state_dim=state_dim,
        lr=lr,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
    )
    episode_rewards = []
    
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 3: TRAINING CONTEXTUAL BANDIT")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  Episodes: {n_episodes}")
        print(f"  Tasks per episode: {n_tasks}")
        print(f"  Learning rate: {lr}")
        print(f"  Initial ε: {epsilon}, decay: {epsilon_decay}, min: {epsilon_min}")
        print("=" * 70 + "\n")
    
    for ep in range(n_episodes):
        env = TaskEnv(n_tasks=n_tasks)
        state = env.reset()
        ep_reward = 0.0
        
        while not env.done:
            available = env.available_actions()
            
            # Select and execute action
            action = agent.select_action(state, available)
            next_state, reward, done, _ = env.step(action)
            
            # Update agent
            agent.update(state, action, reward)
            
            state = next_state if not done else state
            ep_reward += reward
        
        # Decay exploration
        agent.decay_epsilon()
        episode_rewards.append(ep_reward)
        
        # Print progress
        if verbose and (ep + 1) % 200 == 0:
            window = 200
            mean_r = np.mean(episode_rewards[-window:])
            std_r = np.std(episode_rewards[-window:])
            print(f"Episode {ep+1:4d} | Mean: {mean_r:+.3f} (±{std_r:.3f}) | "
                  f"ε: {agent.epsilon:.4f}")
    
    if verbose:
        print("\n" + "=" * 70)
        print("Training complete!")
        print("=" * 70 + "\n")
    
    return agent, np.array(episode_rewards)


def main():
    """Train and save bandit agent."""
    agent, rewards = train_bandit(n_episodes=2000, verbose=True)
    
    # Save model
    save_path = Path("models/bandit.pkl")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({"agent": agent, "rewards": rewards}, f)
    print(f"Saved model to {save_path}\n")
    
    return agent, rewards


if __name__ == "__main__":
    agent, rewards = main()
