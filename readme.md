# Task Prioritization using Contextual Bandits and Reinforcement Learning

> A practical, simulation-based system that learns to prioritize tasks using heuristics, contextual bandits, and deep reinforcement learning.

---

## 1. Project Overview

This project builds a task prioritization agent that selects which task to execute first from a dynamic list, given each task's deadline, duration, and importance. We start with rule-based heuristics as baselines, then train a Contextual Bandit with ε-greedy exploration, and finally extend to a DQN agent for sequential decision-making. No external datasets are required — all tasks are synthetically generated. The goal is to maximize cumulative reward (task utility) while minimizing deadline violations.

---

## 2. Problem Formulation

### State
A fixed-size feature vector representing the current task list (padded to `MAX_TASKS = 5`):

```python
# Per-task features: [time_to_deadline, duration, importance, slack]
# slack = time_to_deadline - duration  (negative = already late)
state = np.array([
    [2.0, 1.0, 0.8, 1.0],   # task 0
    [5.0, 3.0, 0.5, 2.0],   # task 1
    [1.0, 0.5, 1.0, 0.5],   # task 2
    [0.0, 0.0, 0.0, 0.0],   # padding
    [0.0, 0.0, 0.0, 0.0],   # padding
])  # shape: (MAX_TASKS, 4)  →  flatten to (20,) for bandit/DQN input
```

### Action
An integer index `a ∈ {0, 1, ..., MAX_TASKS-1}` representing which task to execute next.

### Reward
```python
def compute_reward(task, current_time):
    finish_time = current_time + task["duration"]
    on_time = finish_time <= task["deadline"]
    if on_time:
        return task["importance"] * 1.0       # positive: importance score
    else:
        lateness = finish_time - task["deadline"]
        return task["importance"] * -lateness  # penalty proportional to lateness
```

### Environment
- At each step: agent observes state, picks a task, executes it (advances `current_time += duration`), receives reward, task is removed from the list.
- Episode ends when all tasks are completed.
- A new episode generates a fresh random task set.

---

## 3. Step-by-Step Implementation Plan

---

### Phase 1 — Environment & Task Generator

**Goal:** Build the simulation engine everything else depends on.

#### Step 1.1 — Task Generator

```python
# src/environment.py
import numpy as np

MAX_TASKS = 5

def generate_tasks(n=MAX_TASKS, seed=None):
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(n):
        duration    = rng.uniform(0.5, 4.0)
        deadline    = rng.uniform(duration + 0.5, duration + 8.0)  # always feasible
        importance  = rng.uniform(0.1, 1.0)
        tasks.append({
            "id":         i,
            "duration":   round(duration, 2),
            "deadline":   round(deadline, 2),
            "importance": round(importance, 2),
        })
    return tasks
```

#### Step 1.2 — State Encoder

```python
def encode_state(tasks, current_time, max_tasks=MAX_TASKS):
    features = []
    for t in tasks:
        ttd   = t["deadline"] - current_time          # time to deadline
        slack = ttd - t["duration"]                   # slack (can be negative)
        features.append([ttd, t["duration"], t["importance"], slack])
    # Pad to max_tasks
    while len(features) < max_tasks:
        features.append([0.0, 0.0, 0.0, 0.0])
    return np.array(features[:max_tasks], dtype=np.float32).flatten()  # shape: (max_tasks*4,)
```

#### Step 1.3 — Task Environment Class

```python
class TaskEnv:
    def __init__(self, n_tasks=MAX_TASKS, seed=None):
        self.n_tasks = n_tasks
        self.seed = seed

    def reset(self):
        self.tasks = generate_tasks(self.n_tasks, self.seed)
        self.current_time = 0.0
        self.done = False
        return encode_state(self.tasks, self.current_time)

    def step(self, action):
        assert not self.done, "Episode is over. Call reset()."
        assert action < len(self.tasks), f"Invalid action {action}, only {len(self.tasks)} tasks left."

        task = self.tasks.pop(action)
        reward = compute_reward(task, self.current_time)
        self.current_time += task["duration"]

        self.done = len(self.tasks) == 0
        next_state = encode_state(self.tasks, self.current_time) if not self.done else None
        info = {"task": task, "current_time": self.current_time}
        return next_state, reward, self.done, info

    def available_actions(self):
        return list(range(len(self.tasks)))
```

**Verify it works:**
```python
env = TaskEnv(n_tasks=3)
state = env.reset()
print("Initial state shape:", state.shape)  # (20,)

while not env.done:
    action = env.available_actions()[0]  # always pick first task
    state, reward, done, info = env.step(action)
    print(f"Reward: {reward:.2f} | Done: {done}")
```

---

### Phase 2 — Heuristic Baselines

**Goal:** Establish performance floors. These will be your comparison benchmarks.

#### Implement all three as selection functions:

```python
# src/heuristics.py

def earliest_deadline_first(tasks, current_time):
    """EDF: pick task with smallest deadline."""
    return min(range(len(tasks)), key=lambda i: tasks[i]["deadline"])

def highest_importance_first(tasks, current_time):
    """HIF: pick task with highest importance."""
    return max(range(len(tasks)), key=lambda i: tasks[i]["importance"])

def shortest_job_first(tasks, current_time):
    """SJF: pick task with smallest duration."""
    return min(range(len(tasks)), key=lambda i: tasks[i]["duration"])

def slack_first(tasks, current_time):
    """Bonus: pick task with least slack (most urgent)."""
    return min(range(len(tasks)), key=lambda i: (tasks[i]["deadline"] - current_time) - tasks[i]["duration"])
```

#### Evaluation runner for heuristics:

```python
# src/evaluate.py
def run_heuristic(policy_fn, n_episodes=200, n_tasks=MAX_TASKS):
    rewards = []
    for ep in range(n_episodes):
        env = TaskEnv(n_tasks=n_tasks, seed=ep)
        env.reset()
        ep_reward = 0
        while not env.done:
            action = policy_fn(env.tasks, env.current_time)
            _, r, _, _ = env.step(action)
            ep_reward += r
        rewards.append(ep_reward)
    return np.array(rewards)

# Run all baselines
results = {
    "EDF":  run_heuristic(earliest_deadline_first),
    "HIF":  run_heuristic(highest_importance_first),
    "SJF":  run_heuristic(shortest_job_first),
    "Slack": run_heuristic(slack_first),
}
for name, r in results.items():
    print(f"{name}: mean={r.mean():.2f}, std={r.std():.2f}")
```

**Expected output:** Numbers showing which heuristic performs best. EDF usually wins on deadline-heavy tasks; HIF wins when deadlines are loose.

---

### Phase 3 — Contextual Bandit (ε-greedy)

**Goal:** Train an agent that learns from interaction using a linear model per action.

#### Design decisions:
- **One linear model per action** (task slot): `Q(s, a) = w_a · φ(s)`
- **Input features** `φ(s)`: the state vector (20-dim), plus bias
- **Update rule**: online gradient step (MSE loss)

#### Step 3.1 — Bandit Agent

```python
# src/bandit.py
import numpy as np

class EpsilonGreedyBandit:
    def __init__(self, n_actions, state_dim, lr=0.01, epsilon=0.2, epsilon_decay=0.995, epsilon_min=0.05):
        self.n_actions    = n_actions
        self.state_dim    = state_dim + 1  # +1 for bias
        self.lr           = lr
        self.epsilon      = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min  = epsilon_min
        # One weight vector per action
        self.weights = np.zeros((n_actions, self.state_dim))

    def _features(self, state):
        return np.append(state, 1.0)  # add bias term

    def predict(self, state, available_actions):
        phi = self._features(state)
        return {a: self.weights[a] @ phi for a in available_actions}

    def select_action(self, state, available_actions):
        if np.random.rand() < self.epsilon:
            return np.random.choice(available_actions)  # explore
        q_values = self.predict(state, available_actions)
        return max(q_values, key=q_values.get)           # exploit

    def update(self, state, action, reward):
        phi    = self._features(state)
        q_pred = self.weights[action] @ phi
        error  = reward - q_pred
        self.weights[action] += self.lr * error * phi   # gradient step

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
```

#### Step 3.2 — Training Loop

```python
# src/train_bandit.py
def train_bandit(n_episodes=2000, n_tasks=MAX_TASKS):
    state_dim = n_tasks * 4
    agent = EpsilonGreedyBandit(n_actions=n_tasks, state_dim=state_dim)
    episode_rewards = []

    for ep in range(n_episodes):
        env = TaskEnv(n_tasks=n_tasks)
        state = env.reset()
        ep_reward = 0

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
            print(f"Episode {ep+1} | Mean Reward: {mean_r:.2f} | ε: {agent.epsilon:.3f}")

    return agent, episode_rewards
```

#### Step 3.3 — Save trained agent

```python
import pickle
agent, rewards = train_bandit()
with open("models/bandit.pkl", "wb") as f:
    pickle.dump(agent, f)
```

---

### Phase 4 — Evaluation

**Goal:** Compare all methods on the same test episodes. No data leakage — use fixed seeds for test episodes.

#### Metrics to compute:

| Metric | Description |
|--------|-------------|
| `mean_reward` | Average episode reward across test episodes |
| `std_reward` | Stability of the policy |
| `deadline_violations` | % of tasks completed after deadline |
| `mean_lateness` | Average delay past deadline (0 if on time) |

#### Evaluation code:

```python
# src/evaluate.py
def evaluate_agent(agent_fn, n_episodes=300, n_tasks=MAX_TASKS, seed_offset=9999):
    """
    agent_fn: callable(state, available_actions) -> action
    Uses seed_offset to ensure test seeds differ from training seeds.
    """
    rewards, violations, latenesses = [], [], []

    for ep in range(n_episodes):
        env = TaskEnv(n_tasks=n_tasks, seed=seed_offset + ep)
        state = env.reset()
        ep_reward, ep_violations, ep_late = 0, 0, 0

        while not env.done:
            action = agent_fn(state, env.available_actions(), env.tasks, env.current_time)
            state, reward, done, info = env.step(action)
            ep_reward += reward
            task = info["task"]
            finish = info["current_time"]
            if finish > task["deadline"]:
                ep_violations += 1
                ep_late += finish - task["deadline"]

        rewards.append(ep_reward)
        violations.append(ep_violations / n_tasks)
        latenesses.append(ep_late)

    return {
        "mean_reward":    np.mean(rewards),
        "std_reward":     np.std(rewards),
        "violation_rate": np.mean(violations),
        "mean_lateness":  np.mean(latenesses),
    }
```

#### Plotting results:

```python
# src/plot_results.py
import matplotlib.pyplot as plt

def plot_comparison(all_results: dict):
    """all_results = {"EDF": {...}, "Bandit": {...}, ...}"""
    names  = list(all_results.keys())
    means  = [all_results[n]["mean_reward"] for n in names]
    stds   = [all_results[n]["std_reward"]  for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Bar chart: mean reward with std error bars
    axes[0].bar(names, means, yerr=stds, capsize=5, color=["#4C72B0","#DD8452","#55A868","#C44E52"])
    axes[0].set_title("Mean Episode Reward (± std)")
    axes[0].set_ylabel("Total Reward")

    # Violation rate
    viols = [all_results[n]["violation_rate"] for n in names]
    axes[1].bar(names, viols, color=["#4C72B0","#DD8452","#55A868","#C44E52"])
    axes[1].set_title("Deadline Violation Rate")
    axes[1].set_ylabel("Fraction of Tasks Late")
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("results/comparison.png", dpi=150)
    plt.show()

def plot_learning_curve(episode_rewards, label="Bandit", window=50):
    smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode="valid")
    plt.figure(figsize=(10, 4))
    plt.plot(smoothed, label=f"{label} (smoothed, w={window})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve")
    plt.legend()
    plt.savefig("results/learning_curve.png", dpi=150)
    plt.show()
```

---

### Phase 5 — DQN Extension (Small Scale)

**Goal:** Replace the linear bandit with a neural network that can capture non-linear patterns. Keep it small and debuggable.

> ⚠️ Keep `MAX_TASKS = 5`, `n_episodes ≤ 3000`, network ≤ 2 hidden layers.

#### Step 5.1 — State modification

The state is already flat `(20,)` — no changes needed. For DQN we also track the number of remaining tasks as a mask to zero out invalid Q-values.

#### Step 5.2 — Network Architecture

```python
# src/dqn.py
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_dim=20, n_actions=5, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)
```

#### Step 5.3 — Replay Buffer

```python
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, mask):
        self.buffer.append((state, action, reward, next_state, done, mask))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
```

#### Step 5.4 — DQN Training Loop

```python
# src/train_dqn.py
import torch.optim as optim
import torch.nn.functional as F

def train_dqn(n_episodes=3000, n_tasks=MAX_TASKS, gamma=0.95, batch_size=64, lr=1e-3):
    state_dim  = n_tasks * 4
    q_net      = QNetwork(state_dim, n_tasks)
    target_net = QNetwork(state_dim, n_tasks)
    target_net.load_state_dict(q_net.state_dict())
    optimizer  = optim.Adam(q_net.parameters(), lr=lr)
    buffer     = ReplayBuffer()

    epsilon, eps_decay, eps_min = 1.0, 0.997, 0.05
    episode_rewards = []
    UPDATE_TARGET_EVERY = 50

    for ep in range(n_episodes):
        env = TaskEnv(n_tasks=n_tasks)
        state = env.reset()
        ep_reward = 0

        while not env.done:
            available = env.available_actions()
            # Action masking: set invalid actions to -inf before argmax
            with torch.no_grad():
                q_vals = q_net(torch.FloatTensor(state)).numpy()
            mask = np.full(n_tasks, -np.inf)
            mask[available] = q_vals[available]

            action = np.random.choice(available) if np.random.rand() < epsilon else int(np.argmax(mask))

            next_state, reward, done, _ = env.step(action)
            next_mask = env.available_actions() if not done else []
            buffer.push(state, action, reward, next_state if not done else np.zeros(state_dim), done, next_mask)
            state     = next_state if not done else state
            ep_reward += reward

            # Train
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                s, a, r, ns, d, _ = zip(*batch)
                s   = torch.FloatTensor(np.array(s))
                a   = torch.LongTensor(a)
                r   = torch.FloatTensor(r)
                ns  = torch.FloatTensor(np.array(ns))
                d   = torch.FloatTensor(d)

                q_current = q_net(s).gather(1, a.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    q_next = target_net(ns).max(1)[0]
                    q_target = r + gamma * q_next * (1 - d)

                loss = F.mse_loss(q_current, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(eps_min, epsilon * eps_decay)
        episode_rewards.append(ep_reward)

        if (ep + 1) % UPDATE_TARGET_EVERY == 0:
            target_net.load_state_dict(q_net.state_dict())

        if (ep + 1) % 500 == 0:
            print(f"Ep {ep+1} | Mean Reward: {np.mean(episode_rewards[-200:]):.2f} | ε: {epsilon:.3f}")

    return q_net, episode_rewards
```

#### Step 5.5 — Save model

```python
torch.save(q_net.state_dict(), "models/dqn.pth")
```

---

## 4. Folder Structure

```
task-prioritization-rl/
├── src/
│   ├── environment.py       # TaskEnv, generate_tasks, encode_state, compute_reward
│   ├── heuristics.py        # EDF, SJF, HIF, slack_first
│   ├── bandit.py            # EpsilonGreedyBandit
│   ├── dqn.py               # QNetwork, ReplayBuffer
│   ├── train_bandit.py      # Training loop for bandit
│   ├── train_dqn.py         # Training loop for DQN
│   └── evaluate.py          # evaluate_agent(), run_heuristic(), plot functions
├── models/
│   ├── bandit.pkl
│   └── dqn.pth
├── results/
│   ├── comparison.png
│   └── learning_curve.png
├── notebooks/
│   └── exploration.ipynb    # Quick experiments, plots
├── tests/
│   └── test_environment.py  # Sanity checks
├── requirements.txt
└── README.md
```

---

## 5. Minimal Tech Stack

```
python>=3.10
numpy
torch>=2.0          # DQN only
matplotlib
jupyter             # Optional, for notebooks
pytest              # Sanity checks
```

**Install:**
```bash
pip install numpy torch matplotlib jupyter pytest
```

No gym, no RLlib, no stable-baselines. Everything is custom and transparent.

---

## 6. Timeline — 2-Week Plan

| Day | Task | Deliverable |
|-----|------|-------------|
| **1** | Set up repo, folder structure, `requirements.txt` | Clean repo skeleton |
| **2** | Implement `generate_tasks`, `encode_state`, `compute_reward` | Working task generator |
| **3** | Implement `TaskEnv` (reset/step), write basic tests | Passing `test_environment.py` |
| **4** | Implement all 4 heuristics + `run_heuristic()` | Printed baseline scores |
| **5** | Implement `EpsilonGreedyBandit` (predict, select, update) | Agent runs without crash |
| **6** | Write bandit training loop, run 2000 episodes | Learning curve (even if noisy) |
| **7** | Build `evaluate_agent()`, unify agent interfaces | Comparable metrics across all methods |
| **8** | Plot comparison bar chart + learning curve | `results/comparison.png` |
| **9** | Implement `QNetwork` + `ReplayBuffer` | Forward pass runs, buffer fills |
| **10** | Write DQN training loop with action masking | DQN trains without errors |
| **11** | Run DQN 3000 episodes, debug reward signal | DQN learning curve visible |
| **12** | Add DQN to evaluation comparison | 5-way comparison table |
| **13** | Write notebook with all results + narrative | `notebooks/exploration.ipynb` done |
| **14** | Clean code, finalize README, push to GitHub | Submission-ready repo |

---

## 7. Common Pitfalls

- **Action index out of bounds after task removal**: Always call `env.available_actions()` before selecting; task list shrinks each step.
- **State dimension mismatch after padding**: Ensure `encode_state` always outputs exactly `(MAX_TASKS * 4,)` even when fewer tasks remain.
- **Bandit not learning**: Check your `lr` (try 0.01–0.001) and confirm `epsilon` is decaying. Print `agent.weights` to check they're changing.
- **DQN reward collapse**: If reward suddenly drops, your target network may be updating too frequently. Increase `UPDATE_TARGET_EVERY` to 100+.
- **Evaluating with training seeds**: Always use `seed_offset` to keep test episodes separate from training. Otherwise your metrics are optimistically biased.
- **Negative reward dominates**: If tasks are frequently late, mean reward will always be negative. This is fine — you're optimizing relative to baselines, not absolute zero.
- **Comparing apples to oranges**: Make sure all methods are evaluated on identical test episodes (same seeds, same `n_tasks`).
- **Overfitting bandit to task order**: The linear bandit maps action index (slot 0, 1, ...) not task identity. Randomize task order in `generate_tasks` so slot 0 isn't always the "easiest" task.

---

## 8. Definition of "Done"

A successful project means **all** of the following are true:

- [ ] `TaskEnv` runs 1000 episodes without errors or index exceptions
- [ ] All 4 heuristics produce printed mean ± std reward over 300 test episodes
- [ ] Bandit trains for 2000 episodes with a visible upward learning curve
- [ ] Bandit outperforms at least one heuristic on mean reward in fair evaluation
- [ ] DQN trains for 3000 episodes and shows improvement over random selection
- [ ] A single `evaluate_agent()` function works for heuristics, bandit, and DQN
- [ ] Bar chart comparing all methods exists in `results/comparison.png`
- [ ] Code is modular: swapping `n_tasks` from 3 to 5 requires changing one constant
- [ ] Notebook tells a story: problem → baselines → bandit → DQN → conclusion
- [ ] All code is committed and pushed to GitHub with a clean commit history

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/task-prioritization-rl
cd task-prioritization-rl
pip install -r requirements.txt

# Run heuristic baselines
python -c "from src.evaluate import *; run_all_baselines()"

# Train bandit
python src/train_bandit.py

# Train DQN
python src/train_dqn.py

# Open notebook
jupyter notebook notebooks/exploration.ipynb
```

---

*Built as a final project. Entirely simulation-based, no external datasets required.*