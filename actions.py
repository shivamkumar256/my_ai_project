"""
actions.py
==========
Discrete action space for the ED Triage environment.

Each action represents a triage decision that the agent (or clinician) makes
for the **head-of-queue** patient at every environment step.
"""

from enum import Enum, auto


class Action(Enum):
    """
    Discrete action space for EDTriageEnv.

    Actions apply to the *current patient* (head of waiting queue):

    ASSIGN_ICU
        Move the patient to an ICU bed.  Correct for severity 1-2.

    ASSIGN_GENERAL
        Move the patient to a general ward bed.  Correct for severity 3.

    ASSIGN_OBSERVATION
        Move the patient to an observation bay.  Correct for severity 4-5.

    HOLD_QUEUE
        Keep the patient in the queue for now.  May be useful when no
        appropriate bed is available; however, the patient deteriorates if
        held too long.

    DISCHARGE
        Release the patient from the ED (e.g. minor illness, walk-in).
        Carries a large penalty if the patient is not stable.
    """

    ASSIGN_ICU = auto()
    ASSIGN_GENERAL = auto()
    ASSIGN_OBSERVATION = auto()
    HOLD_QUEUE = auto()
    DISCHARGE = auto()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def tier_name(self) -> str:
        """
        Map the action to the corresponding ``AssignedTier`` label.
        Returns an empty string for actions that do not set a bed tier.
        """
        mapping = {
            Action.ASSIGN_ICU: "ICU",
            Action.ASSIGN_GENERAL: "General",
            Action.ASSIGN_OBSERVATION: "Observation",
            Action.DISCHARGE: "Discharged",
            Action.HOLD_QUEUE: "Queue",
        }
        return mapping[self]

    @classmethod
    def from_tier(cls, tier: str) -> "Action":
        """
        Reverse-lookup: given a tier string, return the matching Action.

        Raises ``ValueError`` if no action maps to the given tier.
        """
        for action in cls:
            if action.tier_name == tier:
                return action
        raise ValueError(f"No Action maps to tier '{tier}'")

    def __str__(self) -> str:
        return self.name
