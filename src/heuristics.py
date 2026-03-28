"""
Phase 2: Heuristic Baselines

Rule-based task scheduling policies that serve as comparison benchmarks.
Each heuristic uses a simple greedy selection criterion.
"""


def earliest_deadline_first(tasks, current_time):
    """
    EDF: Select task with smallest deadline.
    
    Rationale: Meeting early deadlines prevents cascading penalties from lateness.
    
    Args:
        tasks: list of task dicts
        current_time: current elapsed time (unused for EDF)
    
    Returns:
        int: index of selected task in tasks list
    """
    if not tasks:
        return 0
    return min(range(len(tasks)), key=lambda i: tasks[i]["deadline"])


def highest_importance_first(tasks, current_time):
    """
    HIF: Select task with highest importance.
    
    Rationale: Prioritize valuable tasks to maximize total reward.
    
    Args:
        tasks: list of task dicts
        current_time: current elapsed time (unused for HIF)
    
    Returns:
        int: index of selected task in tasks list
    """
    if not tasks:
        return 0
    return max(range(len(tasks)), key=lambda i: tasks[i]["importance"])


def shortest_job_first(tasks, current_time):
    """
    SJF: Select task with smallest duration.
    
    Rationale: Complete quick tasks first to reduce context switching pressure
    and build momentum on the task list.
    
    Args:
        tasks: list of task dicts
        current_time: current elapsed time (unused for SJF)
    
    Returns:
        int: index of selected task in tasks list
    """
    if not tasks:
        return 0
    return min(range(len(tasks)), key=lambda i: tasks[i]["duration"])


def slack_first(tasks, current_time):
    """
    Slack-First: Select task with least slack (most urgent).
    
    Rationale: Slack = time_to_deadline - duration. Negative slack means
    the task is already impossible to complete on time. Focus on tasks
    with minimal buffer before they become unfeasible.
    
    Args:
        tasks: list of task dicts
        current_time: current elapsed time
    
    Returns:
        int: index of selected task in tasks list
    """
    if not tasks:
        return 0
    
    def slack_score(i):
        t = tasks[i]
        time_to_deadline = t["deadline"] - current_time
        slack = time_to_deadline - t["duration"]
        return slack
    
    return min(range(len(tasks)), key=slack_score)


# Heuristic registry for easy iteration
HEURISTICS = {
    "EDF": earliest_deadline_first,
    "HIF": highest_importance_first,
    "SJF": shortest_job_first,
    "Slack": slack_first,
}
