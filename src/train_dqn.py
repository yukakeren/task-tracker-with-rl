"""
Phase 4: Train DQN Agent

Training loop for Deep Q-Network with:
- State normalization for stable learning
- Double DQN for reduced overestimation
- 1 gradient step per episode (efficient for short episodes)
- Periodic target network updates

Saves trained model to models/dqn.pth
"""

import numpy as np
import torch
from pathlib import Path

from src.environment import TaskEnv, MAX_TASKS
from src.dqn import DQNAgent


def train_dqn(
    n_episodes=3000,
    n_tasks=MAX_TASKS,
    hidden=64,
    lr=1e-3,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.998,
    epsilon_min=0.02,
    buffer_capacity=20_000,
    batch_size=64,
    target_update_freq=50,
    verbose=True,
):
    """
    Train a DQN agent on the task prioritization environment.

    Key design choices:
    - State normalization: computed from 300 random episodes before training
    - 1 gradient step per episode: with only ~5 steps per episode, per-step
      training is wasteful. One update per episode keeps the ratio balanced.
    - Double DQN: online net selects, target net evaluates to reduce bias.

    Args:
        n_episodes: number of training episodes
        n_tasks: tasks per episode
        hidden: hidden layer size
        lr: learning rate
        gamma: discount factor
        epsilon/epsilon_decay/epsilon_min: exploration schedule
        buffer_capacity: replay buffer size
        batch_size: mini-batch size
        target_update_freq: episodes between target network syncs
        verbose: print progress

    Returns:
        agent: trained DQNAgent
        episode_rewards: list of per-episode total rewards
    """
    state_dim = n_tasks * 4

    # Compute normalization statistics from random rollouts
    if verbose:
        print("Computing state normalization statistics...", end=" ", flush=True)
    state_mean, state_std = DQNAgent.compute_normalization_stats(
        n_episodes=300, n_tasks=n_tasks
    )
    if verbose:
        print("done")

    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=n_tasks,
        hidden=hidden,
        lr=lr,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        state_mean=state_mean,
        state_std=state_std,
    )
    episode_rewards = []

    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 4: TRAINING DQN AGENT")
        print("=" * 70)
        print(f"  Episodes: {n_episodes}")
        print(f"  Tasks: {n_tasks} | State dim: {state_dim} | Hidden: {hidden}")
        print(f"  LR: {lr} | gamma: {gamma} | Batch: {batch_size}")
        print(f"  epsilon: {epsilon} -> {epsilon_min} (decay {epsilon_decay})")
        print(f"  Target update every {target_update_freq} episodes")
        print(f"  Buffer capacity: {buffer_capacity}")
        print("=" * 70 + "\n")

    for ep in range(n_episodes):
        env = TaskEnv(n_tasks=n_tasks)
        state = env.reset()
        ep_reward = 0.0

        while not env.done:
            available = env.available_actions()
            action = agent.select_action(state, available)
            next_state, reward, done, _ = env.step(action)
            next_available = env.available_actions() if not done else []

            agent.store_transition(
                state, action, reward,
                next_state if not done else np.zeros(state_dim),
                done, next_available,
            )

            state = next_state if not done else state
            ep_reward += reward

        # One gradient step per episode
        agent.train_step()
        agent.decay_epsilon()
        episode_rewards.append(ep_reward)

        # Periodic target network sync
        if (ep + 1) % target_update_freq == 0:
            agent.update_target_network()

        # Progress logging
        if verbose and (ep + 1) % 500 == 0:
            window = min(200, len(episode_rewards))
            recent = episode_rewards[-window:]
            mean_r = np.mean(recent)
            std_r = np.std(recent)
            mean_loss = np.mean(agent.losses[-500:]) if agent.losses else 0.0
            print(
                f"Episode {ep+1:4d} | "
                f"Mean: {mean_r:+.3f} (+-{std_r:.3f}) | "
                f"eps: {agent.epsilon:.4f} | "
                f"Loss: {mean_loss:.4f} | "
                f"Buffer: {len(agent.buffer)}"
            )

    if verbose:
        print("\n" + "=" * 70)
        print("DQN Training complete!")
        print("=" * 70 + "\n")

    return agent, episode_rewards


def main():
    """Train DQN and save model."""
    agent, rewards = train_dqn(n_episodes=3000, verbose=True)

    # Save model + normalization stats
    save_path = Path("models/dqn.pth")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "q_net": agent.q_net.state_dict(),
            "state_mean": agent.state_mean,
            "state_std": agent.state_std,
        },
        save_path,
    )
    print(f"Saved DQN model to {save_path}\n")

    return agent, rewards


if __name__ == "__main__":
    agent, rewards = main()
