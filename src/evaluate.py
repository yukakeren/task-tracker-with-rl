"""
Phase 2 & 4: Evaluation Framework

Unified evaluation for heuristics, bandit, and DQN agents.
Metrics: mean reward, std, violation rate, mean lateness.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.environment import TaskEnv, MAX_TASKS
from src.heuristics import HEURISTICS


def run_heuristic(policy_fn, n_episodes=300, n_tasks=MAX_TASKS, seed_offset=0):
    """
    Evaluate a heuristic policy over multiple episodes.
    
    Args:
        policy_fn: callable(tasks, current_time) -> action_index
        n_episodes: number of test episodes
        n_tasks: number of tasks per episode
        seed_offset: starting seed for reproducibility
    
    Returns:
        array of episode rewards (shape: (n_episodes,))
    """
    rewards = []
    for ep in range(n_episodes):
        env = TaskEnv(n_tasks=n_tasks, seed=seed_offset + ep)
        state = env.reset()
        ep_reward = 0.0
        
        while not env.done:
            action = policy_fn(env.tasks, env.current_time)
            _, reward, done, _ = env.step(action)
            ep_reward += reward
        
        rewards.append(ep_reward)
    
    return np.array(rewards)


def evaluate_heuristic(policy_fn, n_episodes=300, n_tasks=MAX_TASKS, seed_offset=0):
    """
    Comprehensive evaluation of a heuristic policy.
    
    Args:
        policy_fn: callable(tasks, current_time) -> action_index
        n_episodes: number of test episodes
        n_tasks: number of tasks per episode
        seed_offset: starting seed
    
    Returns:
        dict with keys: mean_reward, std_reward, violation_rate, mean_lateness
    """
    rewards = []
    violations = []
    latenesses = []
    
    for ep in range(n_episodes):
        env = TaskEnv(n_tasks=n_tasks, seed=seed_offset + ep)
        state = env.reset()
        ep_reward = 0.0
        ep_late_tasks = 0
        ep_total_lateness = 0.0
        
        while not env.done:
            action = policy_fn(env.tasks, env.current_time)
            _, reward, done, info = env.step(action)
            ep_reward += reward
            
            task = info["task"]
            finish_time = info["current_time"]
            
            if finish_time > task["deadline"]:
                ep_late_tasks += 1
                ep_total_lateness += finish_time - task["deadline"]
        
        rewards.append(ep_reward)
        violations.append(ep_late_tasks / n_tasks)
        latenesses.append(ep_total_lateness)
    
    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "violation_rate": np.mean(violations),
        "mean_lateness": np.mean(latenesses),
        "rewards": np.array(rewards),  # For plotting learning curves
    }


def evaluate_agent(agent, agent_type="bandit", n_episodes=300, n_tasks=MAX_TASKS, seed_offset=9999):
    """
    Unified evaluation for trained agents (bandit or DQN).
    
    Args:
        agent: trained agent object (EpsilonGreedyBandit or QNetwork)
        agent_type: "bandit" or "dqn"
        n_episodes: number of test episodes
        n_tasks: number of tasks per episode
        seed_offset: starting seed to avoid overlap with training
    
    Returns:
        dict with evaluation metrics
    """
    rewards = []
    violations = []
    latenesses = []
    
    for ep in range(n_episodes):
        env = TaskEnv(n_tasks=n_tasks, seed=seed_offset + ep)
        state = env.reset()
        ep_reward = 0.0
        ep_late_tasks = 0
        ep_total_lateness = 0.0
        
        while not env.done:
            available = env.available_actions()
            
            if agent_type == "bandit":
                # Bandit: use exploitation mode (no exploration)
                action = agent.select_action(state, available)
            elif agent_type == "dqn":
                # DQN: use greedy policy
                import torch
                with torch.no_grad():
                    q_vals = agent(torch.FloatTensor(state)).numpy()
                # Mask invalid actions
                mask = np.full(n_tasks, -np.inf)
                mask[available] = q_vals[available]
                action = int(np.argmax(mask))
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            _, reward, done, info = env.step(action)
            ep_reward += reward
            
            task = info["task"]
            finish_time = info["current_time"]
            
            if finish_time > task["deadline"]:
                ep_late_tasks += 1
                ep_total_lateness += finish_time - task["deadline"]
        
        rewards.append(ep_reward)
        violations.append(ep_late_tasks / n_tasks)
        latenesses.append(ep_total_lateness)
    
    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "violation_rate": np.mean(violations),
        "mean_lateness": np.mean(latenesses),
        "rewards": np.array(rewards),
    }


def run_all_heuristics(n_episodes=300, n_tasks=MAX_TASKS):
    """
    Evaluate all 4 heuristics and print results.
    
    Args:
        n_episodes: number of test episodes per heuristic
        n_tasks: number of tasks per episode
    
    Returns:
        dict: {"heuristic_name": {...metrics...}}
    """
    results = {}
    print("\n" + "=" * 70)
    print("PHASE 2: HEURISTIC BASELINES")
    print("=" * 70 + "\n")
    
    for name, policy_fn in HEURISTICS.items():
        print(f"Evaluating {name}...", end=" ", flush=True)
        metrics = evaluate_heuristic(policy_fn, n_episodes=n_episodes, n_tasks=n_tasks, seed_offset=5000)
        results[name] = metrics
        print(f"✓")
        print(f"  Mean Reward: {metrics['mean_reward']:+.3f} (± {metrics['std_reward']:.3f})")
        print(f"  Violation Rate: {metrics['violation_rate']:.1%}")
        print(f"  Mean Lateness: {metrics['mean_lateness']:.3f}\n")
    
    return results


def plot_comparison(all_results, title="Heuristic Baselines Comparison", save_path=None):
    """
    Plot bar charts comparing methods.
    
    Args:
        all_results: dict of {"method_name": {...metrics...}}
        title: plot title
        save_path: optional path to save figure (e.g., "results/comparison.png")
    """
    names = list(all_results.keys())
    means = [all_results[n]["mean_reward"] for n in names]
    stds = [all_results[n]["std_reward"] for n in names]
    viols = [all_results[n]["violation_rate"] for n in names]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mean reward with error bars
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"][:len(names)]
    axes[0].bar(names, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor="black")
    axes[0].set_title(f"{title}\nMean Episode Reward", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Total Reward", fontsize=11)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    
    # Violation rate
    axes[1].bar(names, viols, color=colors, alpha=0.8, edgecolor="black")
    axes[1].set_title("Deadline Violation Rate", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Fraction of Tasks Late", fontsize=11)
    axes[1].set_ylim([0, 1.0])
    axes[1].grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_learning_curve(episode_rewards, label="Policy", window=50, save_path=None):
    """
    Plot learning curve with smoothing.
    
    Args:
        episode_rewards: array of episode rewards
        label: policy name for legend
        window: smoothing window size
        save_path: optional path to save figure
    """
    smoothed = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
    
    plt.figure(figsize=(12, 5))
    plt.plot(episode_rewards, alpha=0.3, label="Raw", color="gray")
    plt.plot(smoothed, label=f"{label} (smoothed, window={window})", linewidth=2, color="#1f77b4")
    plt.xlabel("Episode", fontsize=11)
    plt.ylabel("Total Episode Reward", fontsize=11)
    plt.title(f"Learning Curve: {label}", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    
    plt.show()


def print_comparison_table(all_results):
    """
    Print formatted comparison table.
    
    Args:
        all_results: dict of {"method_name": {...metrics...}}
    """
    print("\n" + "=" * 100)
    print("COMPARISON TABLE")
    print("=" * 100)
    print(f"{'Method':<15} {'Mean Reward':>15} {'Std Reward':>15} {'Violation %':>15} {'Mean Lateness':>15}")
    print("-" * 100)
    
    for name in all_results:
        m = all_results[name]
        print(f"{name:<15} {m['mean_reward']:>+15.3f} {m['std_reward']:>15.3f} "
              f"{m['violation_rate']*100:>14.1f}% {m['mean_lateness']:>15.3f}")
    
    print("=" * 100 + "\n")
