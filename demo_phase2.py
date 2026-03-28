"""
Phase 2: Heuristic Baselines Demo

Quick test of all 4 heuristic policies.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.environment import TaskEnv
from src.heuristics import HEURISTICS
from src.evaluate import run_all_heuristics, plot_comparison, print_comparison_table


def demo_single_episode():
    """Show one episode of each heuristic."""
    print("=" * 70)
    print("DEMO 1: Single Episode per Heuristic")
    print("=" * 70)
    
    for name, policy_fn in HEURISTICS.items():
        env = TaskEnv(n_tasks=4, seed=42)
        env.reset()
        ep_reward = 0.0
        step = 0
        
        print(f"\n{name}:")
        while not env.done:
            step += 1
            action = policy_fn(env.tasks, env.current_time)
            _, reward, done, info = env.step(action)
            ep_reward += reward
            task = info["task"]
            finish = info["current_time"]
            on_time = "✓" if finish <= task["deadline"] else "✗"
            print(f"  Step {step}: Task {task['id']} {on_time} | Reward: {reward:+.2f}")
        
        print(f"  → Total: {ep_reward:+.2f}")


if __name__ == "__main__":
    demo_single_episode()
    
    # Evaluate all heuristics
    print("\n")
    results = run_all_heuristics(n_episodes=200, n_tasks=5)
    
    # Print table
    print_comparison_table(results)
    
    # Plot
    plot_comparison(results, title="Phase 2: Heuristic Baselines", save_path="results/heuristics_comparison.png")
    
    print("\n" + "=" * 70)
    print("Phase 2 demo complete!")
    print("=" * 70)
