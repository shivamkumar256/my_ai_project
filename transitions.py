"""
transitions.py
==============
Pure state-transition logic for the ED Triage environment.

All functions take an ``EDState`` (immutable) and return a *new* ``EDState``,
keeping the environment side-effect free and easy to test in isolation.
"""

from __future__ import annotations

from typing import List, Tuple

from .actions import Action
from .schemas import BedInventory, EDState, Patient, StaffCount

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DETERIORATION_INTERVAL: int = 3   # sim-steps between each severity worsening
MINUTES_PER_STEP: int = 15        # simulated wall-clock minutes per env step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deduct_bed(beds: BedInventory, tier: str) -> BedInventory:
    """Return a new BedInventory with one bed from *tier* deducted."""
    return beds.model_copy(
        update={
            tier.lower(): max(0, getattr(beds, tier.lower()) - 1)
        }
    )


def _restore_bed(beds: BedInventory, tier: str) -> BedInventory:
    """Return a new BedInventory with one bed of *tier* restored."""
    key = tier.lower()
    return beds.model_copy(update={key: getattr(beds, key) + 1})


def _age_queue(
    waiting: List[Patient],
    step_count: int,
) -> Tuple[List[Patient], int]:
    """
    Age every patient in the waiting queue by one step.

    If a patient's ``wait_time`` is a multiple of DETERIORATION_INTERVAL
    (and non-zero), their severity worsens by 0.5.

    Returns
    -------
    aged_patients : updated patient list
    n_deteriorated : how many patients actually worsened this step
    """
    aged: List[Patient] = []
    n_deteriorated = 0
    for p in waiting:
        new_wait = p.wait_time + 1
        if new_wait > 0 and new_wait % DETERIORATION_INTERVAL == 0:
            p = p.with_deterioration()
            n_deteriorated += 1
        else:
            p = p.with_incremented_wait()
        aged.append(p)
    return aged, n_deteriorated


# ---------------------------------------------------------------------------
# Core transition
# ---------------------------------------------------------------------------

def apply_action(state: EDState, action: Action, target_patient_id: Optional[str] = None) -> EDState:
    """
    Apply *action* to *state* and return the resulting ``EDState``.
    """
    waiting: List[Patient] = list(state.waiting_patients)
    active: List[Patient] = list(state.active_patients)
    beds: BedInventory = state.available_beds

    patient = None
    if target_patient_id:
        idx = next((i for i, p in enumerate(waiting) if p.id == target_patient_id), -1)
        if idx != -1:
            patient = waiting.pop(idx)
        else:
            if action == Action.DISCHARGE:
                # Active discharge bypasses the queue aging completely as it's an immediate action 
                # but standard transitions increment step_count, so we must still do it.
                state = discharge_active_patient(state, target_patient_id)
                waiting = list(state.waiting_patients)
                active = list(state.active_patients)
                beds = state.available_beds
                # Patient remains None so no queue action happens.
    else:
        if waiting:
            patient = waiting.pop(0)

    # ------------------------------------------------------------------
    # Step 1: Act on patient
    # ------------------------------------------------------------------
    if patient is not None:
        if action == Action.HOLD_QUEUE:
            # Put the patient back at the front with incremented wait
            waiting.insert(0, patient.with_incremented_wait())

        elif action == Action.DISCHARGE:
            # Patient leaves the ED (active list not updated; no bed used)
            pass

        else:
            # Bed assignment – only consume a bed if one is available
            tier = action.tier_name  # "ICU" | "General" | "Observation"
            tier_key = tier.lower()
            current_count: int = getattr(beds, tier_key)

            if current_count > 0:
                beds = _deduct_bed(beds, tier)
                active.append(patient.with_tier(tier))  # type: ignore[arg-type]
            else:
                # No bed available → patient goes back to queue head
                waiting.insert(0, patient.with_tier("Queue"))

    # ------------------------------------------------------------------
    # Step 2: Age remaining queue (deterioration)
    # ------------------------------------------------------------------
    new_waiting, _ = _age_queue(waiting, state.step_count + 1)

    # ------------------------------------------------------------------
    # Step 3: Advance clock
    # ------------------------------------------------------------------
    return EDState(
        waiting_patients=new_waiting,
        active_patients=active,
        available_beds=beds,
        staff=state.staff,
        time_elapsed=state.time_elapsed + MINUTES_PER_STEP,
        step_count=state.step_count + 1,
    )


def discharge_active_patient(state: EDState, patient_id: str) -> EDState:
    """
    Discharge a specific *active* patient (e.g. after treatment is complete),
    freeing their bed.

    This is a utility for the environment to call during episode management;
    it is **not** one of the agent's actions.

    Parameters
    ----------
    state:
        Current environment state.
    patient_id:
        ``Patient.id`` of the patient to discharge.

    Returns
    -------
    EDState
        New state with the patient removed and their bed restored.

    Raises
    ------
    ValueError
        If no active patient with the given ID is found.
    """
    active = list(state.active_patients)
    beds = state.available_beds

    target = next((p for p in active if p.id == patient_id), None)
    if target is None:
        raise ValueError(f"No active patient with id='{patient_id}'")

    active.remove(target)
    tier = target.assigned_tier  # e.g. "ICU"
    if tier not in ("Queue", "Discharged"):
        beds = _restore_bed(beds, tier)

    return state.model_copy(
        update={
            "active_patients": active,
            "available_beds": beds,
        }
    )
