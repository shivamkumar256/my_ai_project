"""
ed_triage – OpenEnv-compatible Hospital Emergency Department
Triage & Resource Allocation environment.

Quick start
-----------
>>> from ed_triage import EDTriageEnv, Action
>>> env = EDTriageEnv()
>>> state = env.reset()
>>> next_state, reward, done, info = env.step(Action.ASSIGN_GENERAL)

Phase-2 task configs and grader
--------------------------------
>>> from ed_triage import TASK_EASY, TASK_MEDIUM, TASK_HARD, AgentGrader
>>> env = EDTriageEnv(**TASK_MEDIUM.env_kwargs)
>>> grader = AgentGrader()
>>> score, breakdown = grader.grade(env.episode_log, task="medium")
"""

from .env import EDTriageEnv
from .actions import Action
from .schemas import Patient, EDState, Vitals, BedInventory, StaffCount

# Phase-2: task configs
from .tasks import (
    TaskConfig,
    TASK_EASY,
    TASK_MEDIUM,
    TASK_HARD,
    TASK_REGISTRY,
    get_task,
)

# Phase-2: grader
from .grader import AgentGrader, ScoreBreakdown

# Phase-2: reward helpers
from .reward import apply_fatigue_modifier

# Phase-3: LLM triage helper
from .llm_helper import (
    analyze_complaint,
    chat_with_patient_ai,
    get_autonomous_action,
    hint_badge_html,
    set_api_key,
)

__all__ = [
    # Core
    "EDTriageEnv",
    "Action",
    "Patient",
    "EDState",
    "Vitals",
    "BedInventory",
    "StaffCount",
    # Phase-2: tasks
    "TaskConfig",
    "TASK_EASY",
    "TASK_MEDIUM",
    "TASK_HARD",
    "TASK_REGISTRY",
    "get_task",
    # Phase-2: grader
    "AgentGrader",
    "ScoreBreakdown",
    # Phase-2: reward
    "apply_fatigue_modifier",
    # Phase-3: LLM triage helper
    "analyze_complaint",
    "chat_with_patient_ai",
    "get_autonomous_action",
    "set_api_key",
    "hint_badge_html",
]
__version__ = "0.3.0"
