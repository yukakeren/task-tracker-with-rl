# Task Prioritization using Contextual Bandits and Reinforcement Learning

> A practical, simulation-based system that learns to prioritize tasks using heuristics, contextual bandits, and deep reinforcement learning.

---

## 📊 Project Status

| Component                | Status      | Details                                                          |
| ------------------------ | ----------- | ---------------------------------------------------------------- |
| **Phase 1: Environment** | ✅ Complete | Task generator, state encoding, reward function, `TaskEnv` class |
| **Phase 2: Heuristics**  | ✅ Complete | EDF, HIF, SJF, Slack baselines with unified evaluation framework |
| **Phase 3: Bandit**      | ✅ Complete | Linear ε-greedy contextual bandit with online gradient learning  |
| **Phase 4: DQN**         | ✅ Complete | Neural network agent with replay buffer and target network       |
| **Phase 5: Notebook**    | ⏳ Next     | Comprehensive analysis and visualization                         |

**Quick Test:**

```bash
python tests/test_environment.py   # Phase 1 tests
python demo_phase1.py              # Phase 1 demo
python demo_phase2.py              # Phase 2 heuristic comparison
python demo_phase3.py              # Phase 3 bandit training
python demo_phase4.py              # Phase 4 DQN evaluation
```

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

## ✅ Phase 1 — Environment & Task Generator [COMPLETE]

**Goal:** Build the simulation engine everything else depends on.

**Status:** ✓ All components implemented and tested.

**Deliverables:**

- ✓ [src/environment.py](src/environment.py) – Task generator, state encoder, reward computation, TaskEnv class
- ✓ [tests/test_environment.py](tests/test_environment.py) – 8 unit tests (all passing)
- ✓ [demo_phase1.py](demo_phase1.py) – Demonstration of environment usage
- ✓ 1000+ episode runs without errors
- ✓ State shape validation: always (20,) even with variable task counts

**Key Implementation Details:**

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

---

## ✅ Phase 2 — Heuristic Baselines [COMPLETE]

**Goal:** Establish performance floors. These will be your comparison benchmarks.

**Status:** ✓ All 4 heuristics implemented, evaluated, and comparable.

**Deliverables:**

- ✓ [src/heuristics.py](src/heuristics.py) – EDF, HIF, SJF, Slack heuristics
- ✓ [src/evaluate.py](src/evaluate.py) – Unified evaluation framework for all methods
- ✓ [demo_phase2.py](demo_phase2.py) – Heuristic demonstration and comparison
- ✓ Benchmark results on 200 episodes:
  - **EDF**: Mean Reward = -0.923 ± 3.188 (43.3% violations)
  - **HIF**: Mean Reward = -1.616 ± 2.515 (51.6% violations)
  - **SJF**: Mean Reward = -1.432 ± 2.880 (38.0% violations)
  - **Slack**: Mean Reward = -0.720 ± 3.241 (40.0% violations)

**Key Implementation Details:**

**Heuristic Descriptions:**

- **EDF (Earliest Deadline First)**: Prioritizes tasks by deadline urgency
- **HIF (Highest Importance First)**: Prioritizes high-value tasks
- **SJF (Shortest Job First)**: Completes quick tasks first to reduce queue length
- **Slack**: Prioritizes tasks with minimal slack (most likely to be infeasible)

**Evaluation Framework:**

- `run_heuristic()` – Single heuristic evaluation over episodes
- `evaluate_heuristic()` – Comprehensive metrics (reward, violations, lateness)
- `evaluate_agent()` – Unified evaluation for trained agents (bandit/DQN)
- `plot_comparison()` – Bar charts comparing methods
- `print_comparison_table()` – Formatted results table

---

## ✅ Phase 3 — Contextual Bandit (ε-greedy) [COMPLETE]

**Goal:** Train an agent that learns from interaction using a linear model per action.

**Status:** ✓ Linear bandit implemented, trained for 2000 episodes, and evaluated.

**Deliverables:**

- ✓ [src/bandit.py](src/bandit.py) – `EpsilonGreedyBandit` class with predict/select/update
- ✓ [src/train_bandit.py](src/train_bandit.py) – Training loop (2000 episodes)
- ✓ [demo_phase3.py](demo_phase3.py) – Train, evaluate, and compare to heuristics
- ✓ Training runs successfully with learning curve visualization
- ✓ Model saved to `models/bandit.pkl`

**Phase 3 Results (300 test episodes):**

| Method     | Mean Reward | Std Dev | Violation Rate | Mean Lateness |
| ---------- | ----------- | ------- | -------------- | ------------- |
| **EDF**    | -0.688      | 3.075   | 42.3%          | 4.056         |
| **HIF**    | -1.360      | 2.516   | 50.3%          | 9.029         |
| **SJF**    | -1.281      | 2.861   | 36.9%          | 5.621         |
| **Slack**  | -1.478      | 3.406   | 49.2%          | 5.157         |
| **Bandit** | -3.768      | 3.791   | 50.1%          | 9.355         |

**🔍 Analysis:**

- Bandit currently underperforms heuristics. This is expected in early RL training.
- Linear model may be insufficient for complex task interactions
- Next phase (DQN) will use neural networks for non-linear function approximation
- EDF still best overall; DQN should surpass all by learning non-linear patterns

**Key Implementation Details:**

- **Architecture:** One weight vector per action: `Q(s,a) = w_a · φ(s)`
- **Features:** State vector (20-dim) + bias term (21-dim total)
- **Update Rule:** Online MSE gradient: `w_a := w_a + lr(r - Q(s,a))φ(s)`
- **Exploration:** ε-greedy with decay: `ε := max(0.05, 0.995 * ε)`
- **Learning Rate:** 0.01 (can be tuned for faster/slower convergence)

---

### Phase 3 — Contextual Bandit (ε-greedy)

**Goal:** Train an agent that learns from interaction using a linear model per action.

#### Design decisions:

- **One linear model per action** (task slot): `Q(s, a) = w_a · φ(s)`
- **Input features** `φ(s)`: the state vector (20-dim), plus bias
- **Update rule**: online gradient step (MSE loss)

**Implementation Complete.** See [src/bandit.py](src/bandit.py), [src/train_bandit.py](src/train_bandit.py), and [demo_phase3.py](demo_phase3.py).

---

### Phase 4 — Evaluation

**Goal:** Compare all methods on the same test episodes. No data leakage — use fixed seeds for test episodes.

#### Metrics to compute:

| Metric                | Description                                 |
| --------------------- | ------------------------------------------- |
| `mean_reward`         | Average episode reward across test episodes |
| `std_reward`          | Stability of the policy                     |
| `deadline_violations` | % of tasks completed after deadline         |
| `mean_lateness`       | Average delay past deadline (0 if on time)  |

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

## ✅ Phase 4 — DQN Agent [COMPLETE]

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
│   ├── phase4_comparison.png
│   └── dqn_learning_curve.png
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

| Day    | Task                                                         | Status      |
| ------ | ------------------------------------------------------------ | ----------- |
| **1**  | Set up repo, folder structure, `requirements.txt`            | ✅ Complete |
| **2**  | Implement `generate_tasks`, `encode_state`, `compute_reward` | ✅ Complete |
| **3**  | Implement `TaskEnv` (reset/step), write basic tests          | ✅ Complete |
| **4**  | Implement all 4 heuristics + `run_heuristic()`               | ✅ Complete |
| **5**  | Implement `EpsilonGreedyBandit` (predict, select, update)    | ✅ Complete |
| **6**  | Write bandit training loop, run 2000 episodes                | ✅ Complete |
| **7**  | Build `evaluate_agent()`, unify agent interfaces             | ✅ Complete |
| **8**  | Plot comparison bar chart + learning curve                   | ✅ Complete |
| **9**  | Implement `QNetwork` + `ReplayBuffer`                        | ✅ Complete |
| **10** | Write DQN training loop with action masking                  | ✅ Complete |
| **11** | Run DQN 3000 episodes, debug reward signal                   | ✅ Complete |
| **12** | Add DQN to evaluation comparison                             | ✅ Complete |
| **13** | Write notebook with all results + narrative                  | ⏳ Next     |
| **14** | Clean code, finalize README, push to GitHub                  | ⏳ Next     |

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

- [x] `TaskEnv` runs 1000+ episodes without errors or index exceptions
- [x] All 4 heuristics produce printed mean ± std reward over 300 test episodes
- [x] Bandit trains for 2000 episodes with a visible upward learning curve
- [ ] Bandit outperforms at least one heuristic on mean reward in fair evaluation
- [x] DQN trains for 3000 episodes and shows improvement over random selection
- [x] A single `evaluate_agent()` function works for heuristics, bandit, and DQN (framework ready)
- [x] Bar chart comparing all methods exists in `results/phase3_comparison.png`
- [x] Code is modular: swapping `n_tasks` from 3 to 5 requires changing one constant
- [ ] Notebook tells a story: problem → baselines → bandit → DQN → conclusion
- [ ] All code is committed and pushed to GitHub with a clean commit history

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/task-prioritization-rl
cd task-prioritization-rl
pip install -r requirements.txt

# Phase 1: Verify environment
python tests/test_environment.py
python demo_phase1.py

# Phase 2: Heuristic baselines
python demo_phase2.py

# Phase 3: Train and evaluate contextual bandit
python demo_phase3.py

# Phase 4: Train and evaluate DQN
python demo_phase4.py

# Open notebook
jupyter notebook notebooks/exploration.ipynb
```

---

_Built as a final project. Entirely simulation-based, no external datasets required._
