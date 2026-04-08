"""
env.py
======
EDTriageEnv – OpenEnv-compatible Hospital Emergency Department
Triage & Resource Allocation environment.

Interface mirrors the OpenAI Gym / OpenEnv convention:
    reset()  → EDState
    step(action) → (EDState, float, bool, dict)
    state()  → EDState

The environment is deterministic given a fixed seed; its state is a frozen
Pydantic model (``EDState``), making every snapshot trivially serialisable.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .actions import Action
from .reward import RewardBreakdown, compute_reward
from .schemas import BedInventory, EDState, Patient, StaffCount
from .transitions import DETERIORATION_INTERVAL, MINUTES_PER_STEP, apply_action
from .utils import (
    default_beds,
    default_staff,
    generate_arrival_wave,
    generate_patient,
)


class EDTriageEnv:
    """
    OpenEnv-compatible Hospital Emergency Department simulation.

    The environment models a single ED shift in which an agent makes triage
    and bed-assignment decisions for patients arriving in the waiting queue.

    Episode lifecycle
    -----------------
    * Each episode represents one shift (default: 200 steps ≈ 50 simulated hours).
    * At each step the agent acts on the **head-of-queue** patient.
    * New patients arrive probabilistically every ``arrival_interval`` steps.
    * Active patients are automatically discharged after ``treatment_steps`` steps.
    * The episode ends when ``max_steps`` is reached or the queue+active are empty.

    Parameters
    ----------
    max_steps:
        Maximum number of steps before the episode terminates.
    initial_patients:
        Number of patients pre-loaded into the queue at reset.
    arrival_interval:
        How many steps between probabilistic new arrivals (Poisson-like).
    arrival_probability:
        Probability of a new patient arriving each step.
    treatment_steps:
        Sim-steps an active patient occupies a bed before auto-discharge.
    seed:
        Random seed for reproducibility.
    initial_beds:
        Override the default bed inventory.
    initial_staff:
        Override the default staff counts.

    Examples
    --------
    >>> env = EDTriageEnv(seed=42)
    >>> state = env.reset()
    >>> while True:
    ...     action = Action.ASSIGN_GENERAL   # or from your policy
    ...     state, reward, done, info = env.step(action)
    ...     if done:
    ...         break
    """

    # ------------------------------------------------------------------
    # Metadata (mirrors OpenEnv convention)
    # ------------------------------------------------------------------
    metadata: Dict[str, Any] = {
        "name": "EDTriageEnv",
        "version": "0.2.0",
        "render_modes": ["human", "ansi"],
    }

    def __init__(
        self,
        max_steps: int = 200,
        initial_patients: int = 8,
        arrival_interval: int = 3,
        arrival_probability: float = 0.6,
        treatment_steps: int = 10,
        seed: Optional[int] = None,
        initial_beds: Optional[BedInventory] = None,
        initial_staff: Optional[StaffCount] = None,
        task: Optional[str] = None,
        patient_waves: Optional[Dict[int, int]] = None,
        hidden_severity_patients: int = 0,
    ) -> None:
        self.max_steps = max_steps
        self.initial_patients = initial_patients
        self.arrival_interval = arrival_interval
        self.arrival_probability = arrival_probability
        self.treatment_steps = treatment_steps
        self.initial_beds = initial_beds or default_beds()
        self.initial_staff = initial_staff or default_staff()
        self.task = task
        self.patient_waves = patient_waves or {}
        self.hidden_severity_patients = hidden_severity_patients

        # Store bed_capacity (total max) for overcrowding detection
        # It is fixed from the initial inventory and never changes.
        self._bed_capacity: BedInventory = self.initial_beds

        # Re-triage event schedule if a task config is attached
        self._retriage_schedule: Dict[int, List[int]] = {}  # step -> [queue indices]
        if task is not None:
            self._attach_task(task)

        # Internal RNG (seed-able for reproducibility)
        self._rng = random.Random(seed)
        self._seed = seed

        # Runtime state
        self._state: EDState = self._make_initial_state()
        self._cumulative_reward: float = 0.0
        self._episode_log: List[Dict[str, Any]] = []

        # Track active patient treatment timers: {patient_id: steps_remaining}
        self._treatment_timers: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> EDState:
        """
        Reset the environment to the start of a new episode.

        Returns
        -------
        EDState
            The initial environment state with a fresh waiting queue.
        """
        self._rng = random.Random(self._seed)
        self._state = self._make_initial_state()
        self._cumulative_reward = 0.0
        self._episode_log = []
        self._treatment_timers = {}
        # Rebuild re-triage schedule so it fires fresh each episode
        if self.task is not None:
            self._attach_task(self.task)
        return self._state

    def step(self, action: Action, patient_id: Optional[str] = None) -> Tuple[EDState, float, bool, Dict[str, Any]]:
        """
        Advance the environment by one step.

        Parameters
        ----------
        action:
            One of the five ``Action`` enum values.
        patient_id:
            ID of the patient to act upon. If None, uses head of queue.
            
        Returns
        -------
        next_state : EDState
        reward : float
        done : bool
        info : dict
        """
        current_patient: Optional[Patient] = None
        if patient_id:
            current_patient = next((p for p in self._state.waiting_patients if p.id == patient_id), None)
            if not current_patient:
                current_patient = next((p for p in self._state.active_patients if p.id == patient_id), None)
        else:
            current_patient = self._state.current_patient

        # Force HOLD when queue is empty (nothing to act on)
        if current_patient is None:
            action = Action.HOLD_QUEUE

        # ------------------------------------------------------------------
        # (A) Compute reward BEFORE transition (uses pre-transition state)
        # ------------------------------------------------------------------
        if current_patient is not None:
            # Count total deterioration steps this patient has accumulated
            deterioration_steps = (
                current_patient.wait_time // DETERIORATION_INTERVAL
            )

            # Phase-2: detect context for new reward signals
            # Is any OTHER critical patient waiting in the queue?
            other_queue = self._state.waiting_patients[1:]
            critical_waiting = any(
                p.severity_score >= 4 for p in other_queue
            )

            # Efficient discharge: stable patient frees a bed needed by critical
            freed_for_critical = (
                action == Action.DISCHARGE
                and current_patient.is_stable
                and any(
                    p.severity_score >= 4
                    for p in self._state.waiting_patients
                    if p.id != current_patient.id
                )
            )

            breakdown = compute_reward(
                patient=current_patient,
                action=action,
                beds=self._state.available_beds,
                deterioration_steps=deterioration_steps,
                # Phase-2 context
                step_count=self._state.step_count,
                bed_capacity=self._bed_capacity,
                critical_waiting=critical_waiting,
                freed_for_critical=freed_for_critical,
            )
        else:
            breakdown = RewardBreakdown()

        # ------------------------------------------------------------------
        # (B) Apply action → get next state (pure transition)
        # ------------------------------------------------------------------
        next_state = apply_action(self._state, action, patient_id)

        # ------------------------------------------------------------------
        # (C) Register treatment timers for newly admitted patients
        # ------------------------------------------------------------------
        prev_active_ids = {p.id for p in self._state.active_patients}
        for p in next_state.active_patients:
            if p.id not in prev_active_ids:
                self._treatment_timers[p.id] = self.treatment_steps

        # ------------------------------------------------------------------
        # (D) Auto-discharge treated patients (tick treatment timers)
        # ------------------------------------------------------------------
        next_state = self._tick_treatment_timers(next_state)

        # ------------------------------------------------------------------
        # (E) Probabilistic new patient arrival
        # ------------------------------------------------------------------
        next_state = self._maybe_add_patient(next_state)

        # ------------------------------------------------------------------
        # (E2) Re-triage events (task-driven severity changes)
        # ------------------------------------------------------------------
        next_state = self._apply_retriage(next_state)

        # ------------------------------------------------------------------
        # (F) Update internal state and cumulative reward
        # ------------------------------------------------------------------
        self._state = next_state
        reward = breakdown.total
        self._cumulative_reward += reward

        # ------------------------------------------------------------------
        # (G) Episode termination
        # ------------------------------------------------------------------
        done = self._is_done()

        # ------------------------------------------------------------------
        # (H) Build info dict
        # ------------------------------------------------------------------
        breakdown_dict = {
            "critical_miss": breakdown.critical_miss_penalty,
            "deterioration": breakdown.deterioration_penalty,
            "bed_overflow": breakdown.overflow_penalty,
            "tier_reward": breakdown.tier_reward,
        }

        info: Dict[str, Any] = {
            "step": next_state.step_count,
            "action": action,
            "patient": current_patient,
            "reward_breakdown": breakdown_dict,
            "reward": reward,                     # convenience alias
            "state_snapshot": next_state,         # full frozen state snapshot
            "n_deteriorated": sum(
                1 for p in next_state.waiting_patients
                if p.wait_time > 0 and p.wait_time % DETERIORATION_INTERVAL == 0
            ),
            "cumulative_reward": self._cumulative_reward,
            "done": done,
        }

        self._episode_log.append(info)
        return next_state, reward, done, info

    def state(self) -> EDState:
        """Return the current (most recent) environment state."""
        return self._state

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, mode: str = "human") -> Optional[str]:
        """
        Render the current state.

        Parameters
        ----------
        mode:
            ``"human"`` – prints to stdout and returns None.
            ``"ansi"``  – returns a string without printing.
        """
        output = self._state.summary()
        if mode == "human":
            print(output)
            return None
        return output

    # ------------------------------------------------------------------
    # Episode statistics
    # ------------------------------------------------------------------

    @property
    def cumulative_reward(self) -> float:
        """Total accumulated reward for the current episode."""
        return self._cumulative_reward

    @property
    def episode_log(self) -> List[Dict[str, Any]]:
        """Full step-by-step log for the current episode."""
        return list(self._episode_log)

    def episode_stats(self) -> Dict[str, Any]:
        """
        Summary statistics for the completed (or in-progress) episode.

        Returns
        -------
        dict with keys:
            steps, cumulative_reward, mean_reward, n_correct, n_wrong,
            n_deteriorations, n_discharges_unsafe
        """
        n_correct = sum(
            1
            for entry in self._episode_log
            if (
                entry["patient"] is not None
                and entry["action"] not in (Action.HOLD_QUEUE, Action.DISCHARGE)
                and entry["reward_breakdown"].tier_reward > 0
                and entry["reward_breakdown"].tier_reward
                == entry["reward_breakdown"].tier_reward  # always true; anti-NaN guard
                and entry["action"].tier_name == entry["patient"].ideal_tier
            )
        )
        n_wrong = sum(
            1
            for entry in self._episode_log
            if entry["reward_breakdown"].tier_reward < 0
        )
        n_deteriorations = sum(e["n_deteriorated"] for e in self._episode_log)
        n_unsafe_discharge = sum(
            1
            for entry in self._episode_log
            if (
                entry["action"] == Action.DISCHARGE
                and entry["patient"] is not None
                and not entry["patient"].is_stable
            )
        )
        steps = len(self._episode_log)
        return {
            "steps": steps,
            "cumulative_reward": self._cumulative_reward,
            "mean_reward": self._cumulative_reward / max(steps, 1),
            "n_correct": n_correct,
            "n_wrong": n_wrong,
            "n_deteriorations": n_deteriorations,
            "n_discharges_unsafe": n_unsafe_discharge,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_initial_state(self) -> EDState:
        """Build the initial ``EDState`` for a new episode."""
        patients: List[Patient] = generate_arrival_wave(
            n=self.initial_patients,
            seed=self._rng.randint(0, 2**31),
        )
        return EDState(
            waiting_patients=patients,
            active_patients=[],
            available_beds=self.initial_beds,
            staff=self.initial_staff,
            time_elapsed=0,
            step_count=0,
        )

    def _tick_treatment_timers(self, state: EDState) -> EDState:
        """
        Decrement treatment timers for all active patients.
        Patients whose timer reaches 0 are auto-discharged and their bed freed.
        """
        from .transitions import discharge_active_patient

        to_discharge = [
            pid for pid, t in self._treatment_timers.items() if t <= 1
        ]
        for pid in to_discharge:
            try:
                state = discharge_active_patient(state, pid)
            except ValueError:
                pass  # patient may have already left via another path
            self._treatment_timers.pop(pid, None)

        # Tick remaining timers
        for pid in list(self._treatment_timers.keys()):
            if pid in self._treatment_timers:
                self._treatment_timers[pid] -= 1

        return state

    def _maybe_add_patient(self, state: EDState) -> EDState:
        """Probabilistically or deterministically inject new patients into the waiting queue."""
        updated_queue = list(state.waiting_patients)
        
        # Phase 2: deterministic waves
        if state.step_count in self.patient_waves:
            n_new = self.patient_waves[state.step_count]
            new_patients = generate_arrival_wave(n=n_new, seed=self._rng.randint(0, 2**31))
            
            # Apply hidden severity
            for _ in range(min(self.hidden_severity_patients, len(new_patients))):
                idx = self._rng.randint(0, len(new_patients) - 1)
                p = new_patients[idx]
                mild_complaints = ["mild headache", "slight dizziness", "sore throat", "minor cut"]
                new_patients[idx] = p.model_copy(update={
                    "severity_score": float(self._rng.choice([4, 5])),
                    "chief_complaint": self._rng.choice(mild_complaints)
                })
            
            updated_queue.extend(new_patients)
            self.hidden_severity_patients = 0  # Consume the hidden patients
            
        elif self._rng.random() < self.arrival_probability:
            new_patient = generate_patient(rng=self._rng)
            updated_queue.append(new_patient)
            
        return state.model_copy(update={"waiting_patients": updated_queue})

    def _apply_retriage(self, state: EDState) -> EDState:
        """
        Apply scheduled re-triage events for the current step.

        Re-triage events are defined by ``TaskConfig.retriage_at_steps`` and
        ``retriage_patient_indices``.  When a step matches, the *n*-th patient
        still in the waiting queue has their severity worsened by 1.0
        (simulating hidden deterioration discovered mid-episode).
        """
        current_step = state.step_count
        if current_step not in self._retriage_schedule:
            return state

        queue = list(state.waiting_patients)
        for idx in self._retriage_schedule[current_step]:
            if idx < len(queue):
                p = queue[idx]
                # Severity worsens by 1.0 (more critical), clamped to 1.0
                new_sev = max(1.0, p.severity_score - 1.0)
                queue[idx] = p.model_copy(update={"severity_score": new_sev})

        return state.model_copy(update={"waiting_patients": queue})

    def _attach_task(self, task_name: str) -> None:
        """
        Load a ``TaskConfig`` and build the re-triage event schedule.
        """
        from .tasks import get_task
        cfg = get_task(task_name)
        self._retriage_schedule = {}
        for step, idx in zip(cfg.retriage_at_steps, cfg.retriage_patient_indices):
            self._retriage_schedule.setdefault(step, []).append(idx)

    def _is_done(self) -> bool:
        """Episode ends when max steps reached or ED is fully empty."""
        if self._state.step_count >= self.max_steps:
            return True
        if not self._state.waiting_patients and not self._state.active_patients:
            return True
        return False

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"EDTriageEnv(max_steps={self.max_steps}, "
            f"step={self._state.step_count}, "
            f"queue={self._state.queue_length}, "
            f"active={self._state.occupancy})"
        )
