"""
Phase 1: Environment & Task Generator

Implements the task simulation engine:
- Task generation with random deadlines, durations, importance
- State encoding (flattened task features)
- Reward computation (positive for on-time, negative for late)
- TaskEnv wrapper (reset/step interface)
"""

import numpy as np

MAX_TASKS = 5


def generate_tasks(n=MAX_TASKS, seed=None):
    """
    Generate a random task set.
    
    Args:
        n: number of tasks
        seed: random seed for reproducibility
    
    Returns:
        list of dict, each with keys: "id", "duration", "deadline", "importance"
    """
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(n):
        duration = rng.uniform(0.5, 4.0)
        # Deadline is always feasible: must allow enough time to complete the task
        deadline = rng.uniform(duration + 0.5, duration + 8.0)
        importance = rng.uniform(0.1, 1.0)
        tasks.append({
            "id": i,
            "duration": round(duration, 2),
            "deadline": round(deadline, 2),
            "importance": round(importance, 2),
        })
    return tasks


def encode_state(tasks, current_time, max_tasks=MAX_TASKS):
    """
    Convert task list to flat feature vector.
    
    Per-task features: [time_to_deadline, duration, importance, slack]
    where slack = time_to_deadline - duration (can be negative if already late)
    
    Args:
        tasks: list of task dicts
        current_time: current elapsed time
        max_tasks: pad to this size
    
    Returns:
        np.array of shape (max_tasks * 4,), dtype float32
    """
    features = []
    for t in tasks:
        ttd = t["deadline"] - current_time  # time to deadline
        slack = ttd - t["duration"]  # slack (negative means always late)
        features.append([ttd, t["duration"], t["importance"], slack])
    
    # Pad with zeros to max_tasks
    while len(features) < max_tasks:
        features.append([0.0, 0.0, 0.0, 0.0])
    
    return np.array(features[:max_tasks], dtype=np.float32).flatten()


def compute_reward(task, current_time):
    """
    Compute reward for executing a task.
    
    - Positive reward: importance score if completed on time
    - Negative reward: importance * (-lateness) if completed late
    
    Args:
        task: task dict with "duration", "deadline", "importance"
        current_time: current elapsed time when task starts
    
    Returns:
        float reward
    """
    finish_time = current_time + task["duration"]
    on_time = finish_time <= task["deadline"]
    
    if on_time:
        return task["importance"] * 1.0
    else:
        lateness = finish_time - task["deadline"]
        return task["importance"] * (-lateness)


class TaskEnv:
    """
    Gym-like environment for task scheduling.
    
    Interface:
        - reset(): returns initial state (flat vector)
        - step(action): takes task index, returns (next_state, reward, done, info)
        - available_actions(): returns list of valid task indices
    """
    
    def __init__(self, n_tasks=MAX_TASKS, seed=None):
        """
        Args:
            n_tasks: number of tasks per episode
            seed: seed for task generation (None = random)
        """
        self.n_tasks = n_tasks
        self.seed = seed
        self.tasks = []
        self.current_time = 0.0
        self.done = False
    
    def reset(self):
        """Generate new task set and return initial state."""
        self.tasks = generate_tasks(self.n_tasks, self.seed)
        self.current_time = 0.0
        self.done = False
        return encode_state(self.tasks, self.current_time)
    
    def step(self, action):
        """
        Execute one task step.
        
        Args:
            action: int in [0, len(self.tasks) - 1]
        
        Returns:
            next_state: flat feature vector (or None if done)
            reward: float
            done: bool (all tasks completed)
            info: dict with task info and current_time
        """
        assert not self.done, "Episode is over. Call reset()."
        assert action < len(self.tasks), (
            f"Invalid action {action}, only {len(self.tasks)} tasks available."
        )
        
        # Pop selected task
        task = self.tasks.pop(action)
        
        # Compute reward
        reward = compute_reward(task, self.current_time)
        
        # Advance time
        self.current_time += task["duration"]
        
        # Check if done
        self.done = len(self.tasks) == 0
        
        # Next state
        next_state = encode_state(self.tasks, self.current_time) if not self.done else None
        
        info = {
            "task": task,
            "current_time": self.current_time,
            "reward": reward,
        }
        
        return next_state, reward, self.done, info
    
    def available_actions(self):
        """Return list of valid task indices."""
        return list(range(len(self.tasks)))
