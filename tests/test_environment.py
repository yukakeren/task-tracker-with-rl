"""
Sanity checks for Phase 1 environment.
Run with: pytest tests/test_environment.py -v
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment import (
    generate_tasks,
    encode_state,
    compute_reward,
    TaskEnv,
    MAX_TASKS,
)


def test_generate_tasks():
    """Task generator produces valid tasks."""
    tasks = generate_tasks(n=5, seed=42)
    assert len(tasks) == 5
    for t in tasks:
        assert "id" in t
        assert "duration" in t
        assert "deadline" in t
        assert "importance" in t
        assert 0.5 <= t["duration"] <= 4.0
        assert 1.0 <= t["deadline"] <= 12.0
        assert 0.1 <= t["importance"] <= 1.0
        assert t["deadline"] >= t["duration"]  # Always feasible


def test_encode_state():
    """State encoding produces correct shape."""
    tasks = generate_tasks(n=3)
    state = encode_state(tasks, current_time=0.0)
    assert state.shape == (MAX_TASKS * 4,)
    assert state.dtype == np.float32


def test_encode_state_padding():
    """State is padded to MAX_TASKS even with fewer tasks."""
    tasks = generate_tasks(n=2)
    state = encode_state(tasks, current_time=0.0)
    # Last 2 tasks should be all zeros (padding)
    assert np.allclose(state[-8:], 0.0)


def test_compute_reward_on_time():
    """Reward is positive when task completed on time."""
    task = {
        "duration": 2.0,
        "deadline": 5.0,
        "importance": 0.8,
    }
    reward = compute_reward(task, current_time=0.0)
    assert reward > 0
    assert reward == 0.8  # importance * 1.0


def test_compute_reward_late():
    """Reward is negative when task completed late."""
    task = {
        "duration": 2.0,
        "deadline": 5.0,
        "importance": 0.8,
    }
    reward = compute_reward(task, current_time=4.0)
    # finish_time = 4 + 2 = 6, lateness = 1, reward = 0.8 * (-1) = -0.8
    assert reward < 0
    assert reward == -0.8


def test_task_env_reset():
    """TaskEnv reset returns valid initial state."""
    env = TaskEnv(n_tasks=3, seed=42)
    state = env.reset()
    assert state.shape == (MAX_TASKS * 4,)
    assert not env.done
    assert env.current_time == 0.0
    assert len(env.tasks) == 3


def test_task_env_step():
    """TaskEnv step removes task and updates state."""
    env = TaskEnv(n_tasks=3, seed=42)
    env.reset()
    initial_n_tasks = len(env.tasks)
    
    action = 0
    next_state, reward, done, info = env.step(action)
    
    assert len(env.tasks) == initial_n_tasks - 1
    assert env.current_time > 0.0
    assert next_state.shape == (MAX_TASKS * 4,)
    assert "task" in info
    assert "current_time" in info


def test_task_env_episode():
    """Complete episode without errors."""
    env = TaskEnv(n_tasks=3, seed=42)
    state = env.reset()
    ep_reward = 0.0
    step_count = 0
    
    while not env.done:
        action = env.available_actions()[0]  # Always pick first task
        state, reward, done, info = env.step(action)
        ep_reward += reward
        step_count += 1
    
    assert step_count == 3  # Exactly 3 tasks
    assert env.done
    assert isinstance(ep_reward, float)


def test_available_actions():
    """available_actions returns valid indices."""
    env = TaskEnv(n_tasks=3, seed=42)
    env.reset()
    actions = env.available_actions()
    assert len(actions) == 3
    assert actions == [0, 1, 2]
    
    env.step(0)
    actions = env.available_actions()
    assert len(actions) == 2


if __name__ == "__main__":
    # Quick manual test
    print("Testing environment...")
    test_generate_tasks()
    print("✓ generate_tasks")
    test_encode_state()
    print("✓ encode_state")
    test_compute_reward_on_time()
    print("✓ compute_reward (on time)")
    test_compute_reward_late()
    print("✓ compute_reward (late)")
    test_task_env_reset()
    print("✓ TaskEnv.reset()")
    test_task_env_step()
    print("✓ TaskEnv.step()")
    test_task_env_episode()
    print("✓ Complete episode")
    test_available_actions()
    print("✓ available_actions()")
    print("\nAll tests passed!")
