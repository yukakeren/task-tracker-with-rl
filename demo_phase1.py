"""
Quick verification script for Phase 1.
Demonstrates basic environment usage and task execution.

Run with: python demo_phase1.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.environment import TaskEnv, generate_tasks, encode_state, compute_reward
import numpy as np


def demo_basic_usage():
    """Basic environment setup and single step."""
    print("=" * 60)
    print("DEMO 1: Basic Environment Usage")
    print("=" * 60)
    
    env = TaskEnv(n_tasks=3, seed=42)
    state = env.reset()
    
    print(f"Initial state shape: {state.shape}")
    print(f"Number of tasks: {len(env.tasks)}")
    print(f"Current time: {env.current_time}")
    print(f"Available actions: {env.available_actions()}")
    
    print("\nTasks:")
    for i, task in enumerate(env.tasks):
        print(f"  Task {i}: duration={task['duration']}, deadline={task['deadline']}, "
              f"importance={task['importance']}")


def demo_episode():
    """Run a complete episode with first-task policy."""
    print("\n" + "=" * 60)
    print("DEMO 2: Complete Episode (Always pick first task)")
    print("=" * 60)
    
    env = TaskEnv(n_tasks=4, seed=123)
    state = env.reset()
    
    step = 0
    total_reward = 0.0
    late_tasks = 0
    
    print(f"\nStarting episode with {len(env.tasks)} tasks...\n")
    
    while not env.done:
        step += 1
        action = env.available_actions()[0]  # Always pick first task
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        task = info["task"]
        finish_time = info["current_time"]
        on_time = finish_time <= task["deadline"]
        status = "✓ ON-TIME" if on_time else "✗ LATE"
        
        print(f"Step {step}:")
        print(f"  Task {task['id']}: duration={task['duration']}, "
              f"finish={finish_time:.2f}, deadline={task['deadline']}")
        print(f"  Reward: {reward:+.2f}  {status}")
        
        if not on_time:
            late_tasks += 1
        
        state = next_state
    
    print(f"\n{'─' * 60}")
    print(f"Episode finished!")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Tasks completed late: {late_tasks}/{4}")


def demo_state_encoding():
    """Show state encoding details."""
    print("\n" + "=" * 60)
    print("DEMO 3: State Encoding")
    print("=" * 60)
    
    tasks = generate_tasks(n=3, seed=42)
    current_time = 0.0
    
    print("\nTasks:")
    for t in tasks:
        print(f"  {t}")
    
    state = encode_state(tasks, current_time)
    print(f"\nFlattened state (shape {state.shape}):")
    print(f"  {state}")
    
    # Reshape to see task features
    state_reshaped = state.reshape(5, 4)
    print(f"\nReshaped to (5 tasks, 4 features):")
    print(f"  Columns: [time_to_deadline, duration, importance, slack]")
    for i, row in enumerate(state_reshaped):
        if i < len(tasks):
            print(f"  Task {i}: {row}")
        else:
            print(f"  Padding {i}: {row}")


def demo_random_policy():
    """Compare random policy vs always-first policy."""
    print("\n" + "=" * 60)
    print("DEMO 4: Policy Comparison (100 episodes)")
    print("=" * 60)
    
    n_episodes = 100
    first_task_rewards = []
    random_rewards = []
    
    for ep in range(n_episodes):
        # Always-first policy
        env = TaskEnv(n_tasks=3, seed=1000 + ep)
        env.reset()
        ep_r = 0.0
        while not env.done:
            action = env.available_actions()[0]
            _, r, _, _ = env.step(action)
            ep_r += r
        first_task_rewards.append(ep_r)
        
        # Random policy
        env = TaskEnv(n_tasks=3, seed=1000 + ep)
        env.reset()
        ep_r = 0.0
        while not env.done:
            action = np.random.choice(env.available_actions())
            _, r, _, _ = env.step(action)
            ep_r += r
        random_rewards.append(ep_r)
    
    first_task_rewards = np.array(first_task_rewards)
    random_rewards = np.array(random_rewards)
    
    print(f"\nAlways-First Policy:")
    print(f"  Mean reward: {first_task_rewards.mean():+.2f}")
    print(f"  Std reward:  {first_task_rewards.std():.2f}")
    print(f"  Min/Max:     {first_task_rewards.min():+.2f} / {first_task_rewards.max():+.2f}")
    
    print(f"\nRandom Policy:")
    print(f"  Mean reward: {random_rewards.mean():+.2f}")
    print(f"  Std reward:  {random_rewards.std():.2f}")
    print(f"  Min/Max:     {random_rewards.min():+.2f} / {random_rewards.max():+.2f}")
    
    is_better = "WINS" if first_task_rewards.mean() > random_rewards.mean() else "LOSES"
    diff = first_task_rewards.mean() - random_rewards.mean()
    print(f"\nAlways-First {is_better} by {diff:+.2f}")


if __name__ == "__main__":
    demo_basic_usage()
    demo_episode()
    demo_state_encoding()
    demo_random_policy()
    print("\n" + "=" * 60)
    print("Phase 1 demos complete!")
    print("=" * 60)
