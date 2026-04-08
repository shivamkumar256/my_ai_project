"""
reward.py
=========
Reward function for the ED Triage environment.

All reward logic is isolated here so it can be tuned or swapped without
touching the core environment loop.

Phase-1 Reward table
--------------------
| Situation                              | Reward |
|----------------------------------------|--------|
| Correct tier assignment                |  +10   |
| One tier off (over- or under-triage)   |   +3   |
| Wrong tier (two tiers off)             |  -10   |
| Patient deteriorated in queue (held)   |   -5 / step |
| Bed overflow (no bed available)        |   -8   |
| Discharging an unstable patient        |  -20   |
| Discharging a stable patient correctly |  +5    |
| HOLD_QUEUE when a bed IS available     |   -2   |

Phase-2 Additional Reward Terms
--------------------------------
| Situation                                            | Reward |
|------------------------------------------------------|--------|
| Time-decay: critical patient (sev<=2) waits 1 step   |  -0.5  |
| Overcrowding: any ward type at 100% capacity         |  -3.0  |
| Critical miss: sev<=2 → Observation or Queue         | -25.0  |
| Efficient discharge: stable freed bed for critical   |  +8.0  |
| Staff fatigue (after step 20): reward scaled by 0.85 |   N/A  |
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .actions import Action
    from .schemas import Patient, BedInventory

# ---------------------------------------------------------------------------
# Constants – tweak these without changing any logic
# ---------------------------------------------------------------------------
REWARD_CORRECT_TIER: float = 10.0
REWARD_ONE_OFF: float = 3.0
REWARD_WRONG_TIER: float = -10.0
REWARD_DETERIORATION_PER_STEP: float = -5.0
REWARD_OVERFLOW: float = -8.0
REWARD_DISCHARGE_STABLE: float = 5.0
REWARD_DISCHARGE_UNSTABLE: float = -20.0
REWARD_UNNECESSARY_HOLD: float = -2.0

# Phase-2 constants
REWARD_TIME_DECAY_CRITICAL: float = -0.5   # per step a critical (sev<=2) patient waits
REWARD_OVERCROWDING: float = -3.0          # any ward type at 100% capacity
REWARD_CRITICAL_MISS: float = -25.0        # severity<=2 sent to Observation / held in Queue
REWARD_EFFICIENT_DISCHARGE: float = 8.0    # stable patient discharged while critical waits
STAFF_FATIGUE_STEP_THRESHOLD: int = 20     # step after which fatigue modifier applies
STAFF_FATIGUE_MULTIPLIER: float = 0.85     # reward scale factor when fatigued


# ---------------------------------------------------------------------------
# Tier ordering used to measure how "far off" a decision is
# ---------------------------------------------------------------------------
_TIER_ORDER: dict[str, int] = {
    "ICU": 0,
    "General": 1,
    "Observation": 2,
    "Discharged": 3,  # treated as softest tier for discharge scoring
    "Queue": 4,
}


def _tier_distance(assigned: str, ideal: str) -> int:
    """Return the absolute tier-order distance between two tier labels."""
    return abs(_TIER_ORDER.get(assigned, 4) - _TIER_ORDER.get(ideal, 4))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardBreakdown:
    """
    Detailed breakdown of the reward components for a single step.

    Phase-1 fields
    --------------
    tier_reward, deterioration_penalty, overflow_penalty, discharge_reward,
    hold_penalty

    Phase-2 fields
    --------------
    time_decay_penalty   – applied each step a critical patient waits.
    overcrowding_penalty – applied when any ward is at 100% occupancy.
    critical_miss_penalty – applied when a critical patient is misplaced.
    efficient_discharge_bonus – bonus when a stable patientf rees space for a critical one.
    fatigue_modifier     – multiplicative post-factor applied to the raw total.
    """

    # Phase-1
    tier_reward: float = 0.0
    deterioration_penalty: float = 0.0
    overflow_penalty: float = 0.0
    discharge_reward: float = 0.0
    hold_penalty: float = 0.0

    # Phase-2
    time_decay_penalty: float = 0.0
    overcrowding_penalty: float = 0.0
    critical_miss_penalty: float = 0.0
    efficient_discharge_bonus: float = 0.0
    fatigue_modifier: float = 1.0   # multiplicative; 1.0 = no fatigue

    @property
    def raw_total(self) -> float:
        """Sum of all additive components *before* the fatigue modifier."""
        return (
            self.tier_reward
            + self.deterioration_penalty
            + self.overflow_penalty
            + self.discharge_reward
            + self.hold_penalty
            + self.time_decay_penalty
            + self.overcrowding_penalty
            + self.critical_miss_penalty
            + self.efficient_discharge_bonus
        )

    @property
    def total(self) -> float:
        """Final reward after applying the fatigue multiplier."""
        return round(self.raw_total * self.fatigue_modifier, 4)

    def __str__(self) -> str:
        parts = []
        if self.tier_reward:
            parts.append(f"tier={self.tier_reward:+.1f}")
        if self.deterioration_penalty:
            parts.append(f"detn={self.deterioration_penalty:+.1f}")
        if self.overflow_penalty:
            parts.append(f"ovfl={self.overflow_penalty:+.1f}")
        if self.discharge_reward:
            parts.append(f"disc={self.discharge_reward:+.1f}")
        if self.hold_penalty:
            parts.append(f"hold={self.hold_penalty:+.1f}")
        if self.time_decay_penalty:
            parts.append(f"tdec={self.time_decay_penalty:+.1f}")
        if self.overcrowding_penalty:
            parts.append(f"crwd={self.overcrowding_penalty:+.1f}")
        if self.critical_miss_penalty:
            parts.append(f"cmiss={self.critical_miss_penalty:+.1f}")
        if self.efficient_discharge_bonus:
            parts.append(f"effd={self.efficient_discharge_bonus:+.1f}")
        if self.fatigue_modifier != 1.0:
            parts.append(f"fatigue={self.fatigue_modifier:.2f}x")
        return f"RewardBreakdown({', '.join(parts)}) -> total={self.total:+.1f}"


def compute_reward(
    patient: "Patient",
    action: "Action",
    beds: "BedInventory",
    deterioration_steps: int = 0,
    # ---- Phase-2 keyword-only parameters --------------------------------
    step_count: int = 0,
    bed_capacity: Optional["BedInventory"] = None,
    critical_waiting: bool = False,
    freed_for_critical: bool = False,
) -> RewardBreakdown:
    """
    Compute the composite reward for placing *patient* with *action*.

    Parameters
    ----------
    patient:
        The patient being triaged (head-of-queue).
    action:
        The action chosen by the agent.
    beds:
        Current bed inventory *before* the assignment (available beds).
    deterioration_steps:
        Number of simulation steps the patient has already deteriorated while
        waiting.  Each step adds ``REWARD_DETERIORATION_PER_STEP``.
    step_count : int
        Current episode step index.  Used to apply the staff-fatigue modifier
        when ``step_count >= STAFF_FATIGUE_STEP_THRESHOLD``.
    bed_capacity : BedInventory, optional
        Total (max) bed inventory.  When provided together with *beds*,
        overcrowding detection checks whether any ward is at 100% occupancy.
        If ``None``, overcrowding check is skipped.
    critical_waiting : bool
        ``True`` if at least one critical patient (severity <= 2) is currently
        waiting in the queue (excluding the patient being acted on).  Enables
        the *time-decay* penalty.
    freed_for_critical : bool
        ``True`` when this DISCHARGE action releases a bed that will be taken
        by a waiting critical patient.  Enables the *efficient-discharge*
        bonus.

    Returns
    -------
    RewardBreakdown
        Immutable object with per-component rewards and a ``.total`` property.
    """
    from .actions import Action  # local import to avoid circular deps

    tier_reward = 0.0
    overflow_penalty = 0.0
    discharge_reward = 0.0
    hold_penalty = 0.0

    # ------------------------------------------------------------------
    # Phase-1: Deterioration penalty
    # ------------------------------------------------------------------
    deterioration_penalty = deterioration_steps * REWARD_DETERIORATION_PER_STEP

    # ------------------------------------------------------------------
    # Phase-2: Time-decay for any critical patient waiting in queue
    # ------------------------------------------------------------------
    time_decay_penalty = (
        REWARD_TIME_DECAY_CRITICAL if critical_waiting else 0.0
    )

    # ------------------------------------------------------------------
    # Phase-2: Overcrowding penalty
    # ------------------------------------------------------------------
    overcrowding_penalty = 0.0
    if bed_capacity is not None:
        if (
            (bed_capacity.icu > 0 and beds.icu == 0)
            or (bed_capacity.general > 0 and beds.general == 0)
            or (bed_capacity.observation > 0 and beds.observation == 0)
        ):
            overcrowding_penalty = REWARD_OVERCROWDING

    # ------------------------------------------------------------------
    # Phase-1: Action-specific scoring
    # ------------------------------------------------------------------
    critical_miss_penalty = 0.0
    efficient_discharge_bonus = 0.0

    if action == Action.HOLD_QUEUE:
        ideal = patient.ideal_tier
        bed_available = {
            "ICU": beds.icu > 0,
            "General": beds.general > 0,
            "Observation": beds.observation > 0,
        }.get(ideal, False)
        if bed_available:
            hold_penalty = REWARD_UNNECESSARY_HOLD

    elif action == Action.DISCHARGE:
        if patient.is_stable:
            discharge_reward = REWARD_DISCHARGE_STABLE
        else:
            discharge_reward = REWARD_DISCHARGE_UNSTABLE

        # Phase-2: Efficient discharge bonus
        if freed_for_critical and patient.is_stable:
            efficient_discharge_bonus = REWARD_EFFICIENT_DISCHARGE

    else:
        # Bed assignment
        assigned_tier = action.tier_name
        ideal_tier = patient.ideal_tier

        bed_counts = {
            "ICU": beds.icu,
            "General": beds.general,
            "Observation": beds.observation,
        }
        if bed_counts.get(assigned_tier, 0) <= 0:
            overflow_penalty = REWARD_OVERFLOW

        distance = _tier_distance(assigned_tier, ideal_tier)
        if distance == 0:
            tier_reward = REWARD_CORRECT_TIER
        elif distance == 1:
            tier_reward = REWARD_ONE_OFF
        else:
            tier_reward = REWARD_WRONG_TIER

        # Phase-2: Critical miss — severity 5 sent anywhere except ICU
        if patient.severity_score == 5 and action in (Action.HOLD_QUEUE, Action.ASSIGN_OBSERVATION):
            critical_miss_penalty = REWARD_CRITICAL_MISS

    # ------------------------------------------------------------------
    # Phase-2: Staff fatigue modifier
    # ------------------------------------------------------------------
    fatigue_modifier = (
        STAFF_FATIGUE_MULTIPLIER
        if step_count >= STAFF_FATIGUE_STEP_THRESHOLD
        else 1.0
    )

    return RewardBreakdown(
        tier_reward=tier_reward,
        deterioration_penalty=deterioration_penalty,
        overflow_penalty=overflow_penalty,
        discharge_reward=discharge_reward,
        hold_penalty=hold_penalty,
        time_decay_penalty=time_decay_penalty,
        overcrowding_penalty=overcrowding_penalty,
        critical_miss_penalty=critical_miss_penalty,
        efficient_discharge_bonus=efficient_discharge_bonus,
        fatigue_modifier=fatigue_modifier,
    )


# ---------------------------------------------------------------------------
# Phase-2: Public helper — apply fatigue modifier post-hoc
# ---------------------------------------------------------------------------

def apply_fatigue_modifier(
    breakdown: RewardBreakdown,
    step_count: int,
) -> RewardBreakdown:
    """
    Return a *new* ``RewardBreakdown`` with the fatigue modifier updated
    based on *step_count*.

    Useful when you want to compute the base reward first and apply fatigue
    as a separate post-processing step (e.g. in custom wrappers).

    Parameters
    ----------
    breakdown:
        An existing (possibly fatigue-free) ``RewardBreakdown``.
    step_count:
        Current step index.  If ``>= STAFF_FATIGUE_STEP_THRESHOLD``, the
        multiplier is set to ``STAFF_FATIGUE_MULTIPLIER``; otherwise 1.0.

    Returns
    -------
    RewardBreakdown
    """
    modifier = (
        STAFF_FATIGUE_MULTIPLIER
        if step_count >= STAFF_FATIGUE_STEP_THRESHOLD
        else 1.0
    )
    from dataclasses import replace as _dc_replace
    return _dc_replace(breakdown, fatigue_modifier=modifier)
