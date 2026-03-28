"""
Phase 3: Contextual Bandit Demo

Train bandit, evaluate it, and compare against heuristics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from src.environment import TaskEnv, MAX_TASKS
from src.bandit import EpsilonGreedyBandit
from src.heuristics import HEURISTICS
from src.evaluate import (
    evaluate_heuristic,
    evaluate_agent,
    plot_comparison,
    print_comparison_table,
    plot_learning_curve,
)


def demo_single_bandit_episode():
    """Show one training episode of bandit learning."""
    print("=" * 70)
    print("DEMO 1: Single Bandit Training Episode")
    print("=" * 70)
    
    state_dim = MAX_TASKS * 4
    agent = EpsilonGreedyBandit(n_actions=MAX_TASKS, state_dim=state_dim)
    
    env = TaskEnv(n_tasks=4, seed=42)
    state = env.reset()
    ep_reward = 0.0
    step = 0
    
    print(f"\nInitial ε: {agent.epsilon:.3f}\n")
    
    while not env.done:
        step += 1
        available = env.available_actions()
        
        # Select action
        action = agent.select_action(state, available)
        
        # Execute
        next_state, reward, done, info = env.step(action)
        
        # Update
        agent.update(state, action, reward)
        
        task = info["task"]
        finish = info["current_time"]
        on_time = "✓" if finish <= task["deadline"] else "✗"
        
        print(f"Step {step}: Task {task['id']} {on_time} | "
              f"Reward: {reward:+.2f} | Q-pred was: ~{agent.weights[action] @ np.append(state, 1.0):.2f}")
        
        state = next_state if not done else state
        ep_reward += reward
    
    print(f"\nTotal reward: {ep_reward:+.2f}")
    print(f"Final weight norms: {agent.get_weight_norms()}")


def train_and_evaluate_bandit(n_episodes=2000):
    """Train bandit and evaluate on test set."""
    print("\n" + "=" * 70)
    print("TRAINING CONTEXTUAL BANDIT (2000 episodes)")
    print("=" * 70 + "\n")
    
    state_dim = MAX_TASKS * 4
    agent = EpsilonGreedyBandit(
        n_actions=MAX_TASKS,
        state_dim=state_dim,
        lr=0.01,
        epsilon=0.2,
        epsilon_decay=0.995,
        epsilon_min=0.05,
    )
    episode_rewards = []
    
    for ep in range(n_episodes):
        env = TaskEnv(n_tasks=MAX_TASKS)
        state = env.reset()
        ep_reward = 0.0
        
        while not env.done:
            available = env.available_actions()
            action = agent.select_action(state, available)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward)
            state = next_state if not done else state
            ep_reward += reward
        
        agent.decay_epsilon()
        episode_rewards.append(ep_reward)
        
        if (ep + 1) % 200 == 0:
            mean_r = np.mean(episode_rewards[-200:])
            print(f"Episode {ep+1:4d} | Mean: {mean_r:+.3f} | ε: {agent.epsilon:.4f}")
    
    print("✓ Training complete\n")
    
    # Evaluate bandit
    print("Evaluating bandit on test set (300 episodes)...", end=" ", flush=True)
    bandit_metrics = evaluate_agent(agent, agent_type="bandit", n_episodes=300, seed_offset=9999)
    print("✓")
    print(f"  Mean Reward: {bandit_metrics['mean_reward']:+.3f} (± {bandit_metrics['std_reward']:.3f})")
    print(f"  Violation Rate: {bandit_metrics['violation_rate']:.1%}")
    print(f"  Mean Lateness: {bandit_metrics['mean_lateness']:.3f}\n")
    
    return agent, episode_rewards, bandit_metrics


def full_comparison():
    """Train bandit and compare all methods."""
    # Get heuristic results
    print("=" * 70)
    print("PHASE 3: BANDIT vs HEURISTIC BASELINES")
    print("=" * 70 + "\n")
    
    print("Evaluating heuristic baselines...")
    heuristic_results = {}
    for name, policy_fn in HEURISTICS.items():
        print(f"  {name}...", end=" ", flush=True)
        metrics = evaluate_heuristic(policy_fn, n_episodes=300, seed_offset=5000)
        heuristic_results[name] = metrics
        print("✓")
    
    # Train and evaluate bandit
    agent, rewards, bandit_metrics = train_and_evaluate_bandit(n_episodes=2000)
    
    # Combine results
    all_results = {**heuristic_results, "Bandit": bandit_metrics}
    
    print_comparison_table(all_results)
    
    # Plots
    print("Generating plots...")
    plot_comparison(all_results, title="Phase 3: Bandit vs Heuristics", save_path="results/phase3_comparison.png")
    plot_learning_curve(rewards, label="Bandit", window=100, save_path="results/bandit_learning_curve.png")
    
    return agent, rewards, all_results


if __name__ == "__main__":
    demo_single_bandit_episode()
    print("\n")
    agent, rewards, results = full_comparison()
    print("\nPhase 3 demo complete!")
