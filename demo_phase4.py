"""
Phase 4: DQN Demo

Train DQN, evaluate it, and compare against heuristics and bandit.

Usage: python demo_phase4.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

from src.environment import TaskEnv, MAX_TASKS
from src.dqn import DQNAgent
from src.train_dqn import train_dqn
from src.bandit import EpsilonGreedyBandit
from src.heuristics import HEURISTICS
from src.evaluate import (
    evaluate_heuristic,
    evaluate_agent,
    plot_comparison,
    print_comparison_table,
    plot_learning_curve,
)


def demo_single_dqn_episode():
    """Show one episode with an untrained DQN (random weights)."""
    print("=" * 70)
    print("DEMO 1: Single DQN Episode (untrained)")
    print("=" * 70)

    state_dim = MAX_TASKS * 4
    agent = DQNAgent(state_dim=state_dim, n_actions=MAX_TASKS)

    env = TaskEnv(n_tasks=4, seed=42)
    state = env.reset()
    ep_reward = 0.0
    step = 0

    print(f"\nInitial epsilon: {agent.epsilon:.3f}\n")

    while not env.done:
        step += 1
        available = env.available_actions()
        action = agent.select_action(state, available)

        next_state, reward, done, info = env.step(action)
        task = info["task"]
        finish = info["current_time"]
        on_time = "✓" if finish <= task["deadline"] else "✗"

        print(
            f"Step {step}: Task {task['id']} {on_time} | "
            f"Reward: {reward:+.2f}"
        )

        state = next_state if not done else state
        ep_reward += reward

    print(f"\nTotal reward: {ep_reward:+.2f}")


def train_bandit_baseline():
    """Train bandit for fair comparison."""
    state_dim = MAX_TASKS * 4
    bandit = EpsilonGreedyBandit(
        n_actions=MAX_TASKS,
        state_dim=state_dim,
        lr=0.01,
        epsilon=0.2,
        epsilon_decay=0.995,
        epsilon_min=0.05,
    )
    for ep in range(2000):
        env = TaskEnv(n_tasks=MAX_TASKS)
        state = env.reset()
        while not env.done:
            available = env.available_actions()
            action = bandit.select_action(state, available)
            next_state, reward, done, _ = env.step(action)
            bandit.update(state, action, reward)
            state = next_state if not done else state
        bandit.decay_epsilon()
    return bandit


def full_comparison():
    """Train DQN and compare all methods."""
    print("\n" + "=" * 70)
    print("PHASE 4: DQN vs HEURISTICS vs BANDIT")
    print("=" * 70 + "\n")

    # --- Evaluate heuristic baselines ---
    print("Evaluating heuristic baselines...")
    heuristic_results = {}
    for name, policy_fn in HEURISTICS.items():
        print(f"  {name}...", end=" ", flush=True)
        metrics = evaluate_heuristic(policy_fn, n_episodes=300, seed_offset=5000)
        heuristic_results[name] = metrics
        print("✓")

    # --- Train and evaluate bandit ---
    print("\nTraining contextual bandit (2000 episodes)...", end=" ", flush=True)
    bandit = train_bandit_baseline()
    print("done")
    print("  Evaluating bandit...", end=" ", flush=True)
    bandit_metrics = evaluate_agent(
        bandit, agent_type="bandit", n_episodes=300, seed_offset=9999
    )
    print("✓")

    # --- Train and evaluate DQN ---
    dqn_agent, dqn_rewards = train_dqn(n_episodes=3000, verbose=True)

    print("Evaluating DQN on test set (300 episodes)...", end=" ", flush=True)
    dqn_metrics = evaluate_agent(
        dqn_agent, agent_type="dqn", n_episodes=300, seed_offset=9999
    )
    print("✓")
    print(f"  Mean Reward: {dqn_metrics['mean_reward']:+.3f} (± {dqn_metrics['std_reward']:.3f})")
    print(f"  Violation Rate: {dqn_metrics['violation_rate']:.1%}")
    print(f"  Mean Lateness: {dqn_metrics['mean_lateness']:.3f}")

    # --- Combine and compare ---
    all_results = {
        **heuristic_results,
        "Bandit": bandit_metrics,
        "DQN": dqn_metrics,
    }

    print_comparison_table(all_results)

    # Check success criteria
    edf_reward = heuristic_results["EDF"]["mean_reward"]
    dqn_reward = dqn_metrics["mean_reward"]
    print(f"{'─' * 70}")
    if dqn_reward > edf_reward:
        print(f"  ✓ SUCCESS: DQN ({dqn_reward:+.3f}) outperforms EDF ({edf_reward:+.3f})")
        print(f"    Improvement: {dqn_reward - edf_reward:+.3f}")
    else:
        gap = edf_reward - dqn_reward
        print(f"  ✗ DQN ({dqn_reward:+.3f}) is {gap:.3f} behind EDF ({edf_reward:+.3f})")
        print(f"    Consider: more episodes, lower LR, or tuning target_update_freq")
    print(f"{'─' * 70}\n")

    # --- Generate plots ---
    print("Generating plots...")
    Path("results").mkdir(parents=True, exist_ok=True)

    plot_comparison(
        all_results,
        title="Phase 4: DQN vs Heuristics vs Bandit",
        save_path="results/phase4_comparison.png",
    )
    plot_learning_curve(
        dqn_rewards,
        label="DQN",
        window=100,
        save_path="results/dqn_learning_curve.png",
    )

    # Save model
    Path("models").mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "q_net": dqn_agent.q_net.state_dict(),
            "state_mean": dqn_agent.state_mean,
            "state_std": dqn_agent.state_std,
        },
        "models/dqn.pth",
    )
    print("Saved DQN model to models/dqn.pth")

    return dqn_agent, dqn_rewards, all_results


if __name__ == "__main__":
    demo_single_dqn_episode()
    print("\n")
    agent, rewards, results = full_comparison()
    print("\nPhase 4 demo complete!")
