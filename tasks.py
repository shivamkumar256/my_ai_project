"""
tasks.py
========
Predefined task configurations for the ED Triage environment.

Each ``TaskConfig`` is a frozen dataclass that bundles everything needed to
instantiate an ``EDTriageEnv`` for a specific difficulty level and to grade
a completed episode via ``AgentGrader``.

Usage
-----
>>> from ed_triage.tasks import TASK_EASY, TASK_MEDIUM, TASK_HARD
>>> from ed_triage import EDTriageEnv
>>> env = EDTriageEnv(**TASK_EASY.env_kwargs)
>>> state = env.reset()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# TaskConfig dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskConfig:
    """
    Complete specification for one difficulty level of the ED Triage task.

    Attributes
    ----------
    name : str
        Human-readable identifier (``"easy"`` / ``"medium"`` / ``"hard"``).
    description : str
        Short prose description of the scenario.
    env_kwargs : dict
        Keyword arguments forwarded verbatim to ``EDTriageEnv(**env_kwargs)``.
    success_criteria : dict
        Named thresholds used by ``AgentGrader`` to decide pass/fail per
        dimension.  Keys depend on the task; see individual instances below.
    score_weights : dict[str, float]
        Per-dimension weight used when computing the final normalised grade.
        Weights must sum to 1.0.
    fatigue_from_step : int
        Step at which staff fatigue becomes active (0 = always on).
    retriage_patient_indices : list[int]
        Indices into the waiting queue of patients whose severity changes
        mid-episode (used by ``EDTriageEnv`` when ``task`` is provided).
    retriage_at_steps : list[int]
        Parallel list of step numbers at which each retriage event fires.
    seed : int or None
        Scenario seed for reproducibility.
    """

    name: str
    description: str
    env_kwargs: Dict[str, Any]
    success_criteria: Dict[str, Any]
    score_weights: Dict[str, float]
    fatigue_from_step: int = 20
    retriage_patient_indices: List[int] = field(default_factory=list)
    retriage_at_steps: List[int] = field(default_factory=list)
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        total = sum(self.score_weights.values())
        if not (0.999 < total < 1.001):
            raise ValueError(
                f"TaskConfig '{self.name}': score_weights must sum to 1.0, got {total:.4f}"
            )
        if len(self.retriage_patient_indices) != len(self.retriage_at_steps):
            raise ValueError(
                f"TaskConfig '{self.name}': retriage_patient_indices and "
                f"retriage_at_steps must have the same length."
            )


# ---------------------------------------------------------------------------
# Bed inventory factories
# ---------------------------------------------------------------------------

def _make_easy_beds():
    """4 ICU / 4 General / 2 Observation = 10 total."""
    from .schemas import BedInventory
    return BedInventory(icu=4, general=4, observation=2)


def _make_medium_beds():
    """2 ICU / 4 General / 2 Observation = 8 total."""
    from .schemas import BedInventory
    return BedInventory(icu=2, general=4, observation=2)


def _make_hard_beds():
    """
    ICU starts at 80% full:
      Total ICU = 5, 4 already occupied -> 1 available.
      General = 10 available, Observation = 5 available.
    """
    from .schemas import BedInventory
    return BedInventory(icu=1, general=10, observation=5)


# ---------------------------------------------------------------------------
# TASK_EASY
# ---------------------------------------------------------------------------

TASK_EASY = TaskConfig(
    name="easy",
    description=(
        "5 patients with unambiguous vitals, 10 beds total. "
        "All patients map clearly to a single correct tier. "
        "Success = 100% correct tier assignment."
    ),
    env_kwargs=dict(
        max_steps=30,
        initial_patients=5,
        arrival_probability=0.0,       # no new arrivals - fixed cohort
        treatment_steps=20,            # patients stay long enough to be handled
        initial_beds=None,             # filled in below after bed factory is defined
        seed=1001,
    ),
    success_criteria={
        "tier_accuracy": 1.0,          # 100% of placements must be correct
    },
    score_weights={
        "accuracy": 1.0,               # only correctness matters
    },
    fatigue_from_step=9999,            # fatigue effectively disabled for Easy
    seed=1001,
)
# Inject the bed inventory now that _make_easy_beds is defined
TASK_EASY = TaskConfig(
    name=TASK_EASY.name,
    description=TASK_EASY.description,
    env_kwargs={**TASK_EASY.env_kwargs, "initial_beds": _make_easy_beds()},
    success_criteria=TASK_EASY.success_criteria,
    score_weights=TASK_EASY.score_weights,
    fatigue_from_step=TASK_EASY.fatigue_from_step,
    retriage_patient_indices=TASK_EASY.retriage_patient_indices,
    retriage_at_steps=TASK_EASY.retriage_at_steps,
    seed=TASK_EASY.seed,
)


# ---------------------------------------------------------------------------
# TASK_MEDIUM
# ---------------------------------------------------------------------------

TASK_MEDIUM = TaskConfig(
    name="medium",
    description=(
        "15 patients arriving in 3 waves (every 5 steps), 8 beds total. "
        "2 critical patients present with deceptively mild-looking complaints. "
        "Success = zero critical misses + avg wait < 6 steps."
    ),
    env_kwargs=dict(
        initial_patients=5,
        initial_beds=_make_medium_beds(),
        patient_waves={
            5: 5,    # 5 more patients arrive at step 5
            10: 5,   # 5 more patients arrive at step 10
        },
        hidden_severity_patients=2,
        max_steps=30,
    ),
    success_criteria={
        "zero_critical_misses": True,
        "max_avg_wait_steps": 6.0,
    },
    score_weights={
        "throughput": 0.60,
        "critical_safety": 0.40,
    },
)


# ---------------------------------------------------------------------------
# TASK_HARD
# ---------------------------------------------------------------------------

TASK_HARD = TaskConfig(
    name="hard",
    description=(
        "40 patients over 30 sim-steps (mass casualty). "
        "ICU starts at 80% full. Staff fatigue active from step 1. "
        "5 patients change severity mid-episode (re-triage events). "
        "Success = mortality_rate < 5% + resource utilization > 75%."
    ),
    env_kwargs=dict(
        max_steps=30,
        initial_patients=12,
        arrival_probability=0.9,        # high surge
        treatment_steps=5,              # rapid turnover required
        initial_beds=_make_hard_beds(),
        seed=3003,
    ),
    success_criteria={
        "max_mortality_rate": 0.05,     # < 5% of critical patients mishandled
        "min_utilization": 0.75,        # > 75% bed utilization
    },
    score_weights={
        "mortality": 0.50,
        "utilization": 0.30,
        "speed": 0.20,
    },
    fatigue_from_step=0,               # fatigue active from the very first step
    retriage_patient_indices=[0, 2, 4, 6, 8],    # patients that change severity
    retriage_at_steps=[5, 8, 12, 16, 22],         # when each re-triage fires
    seed=3003,
)


# ---------------------------------------------------------------------------
# Registry and lookup helper
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, TaskConfig] = {
    "easy": TASK_EASY,
    "medium": TASK_MEDIUM,
    "hard": TASK_HARD,
}


def get_task(name: str) -> TaskConfig:
    """
    Retrieve a ``TaskConfig`` by name (case-insensitive).

    Parameters
    ----------
    name : str
        One of ``"easy"``, ``"medium"``, ``"hard"``.

    Returns
    -------
    TaskConfig

    Raises
    ------
    KeyError
        If *name* is not registered.
    """
    key = name.strip().lower()
    if key not in TASK_REGISTRY:
        raise KeyError(
            f"Unknown task '{name}'. Available: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[key]
