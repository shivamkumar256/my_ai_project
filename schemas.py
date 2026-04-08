"""
schemas.py
==========
Pydantic data models for the ED Triage environment.

All models are immutable by default (frozen=True) so that environment state
transitions always produce *new* objects, making the data flow explicit and
safe for serialisation / logging.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Severity tier type alias
# ---------------------------------------------------------------------------
SeverityScore = Literal[1, 2, 3, 4, 5]
"""
ESI-inspired triage levels:
  1 – Immediate / life-threatening
  2 – Emergent
  3 – Urgent
  4 – Less urgent
  5 – Non-urgent
"""

AssignedTier = Literal["ICU", "General", "Observation", "Queue", "Discharged"]


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class Vitals(BaseModel, frozen=True):
    """Patient vital signs captured at arrival / last assessment."""

    bp_systolic: int = Field(..., ge=60, le=220, description="Systolic blood pressure (mmHg)")
    bp_diastolic: int = Field(..., ge=40, le=140, description="Diastolic blood pressure (mmHg)")
    hr: int = Field(..., ge=20, le=250, description="Heart rate (bpm)")
    spo2: float = Field(..., ge=50.0, le=100.0, description="Oxygen saturation (%)")
    temp: float = Field(..., ge=32.0, le=42.5, description="Body temperature (°C)")

    @property
    def bp(self) -> str:
        """Human-readable blood pressure string."""
        return f"{self.bp_systolic}/{self.bp_diastolic}"

    def to_dict(self) -> dict:
        return {
            "bp": self.bp,
            "hr": self.hr,
            "spo2": self.spo2,
            "temp": self.temp,
        }


class BedInventory(BaseModel, frozen=True):
    """Available (unoccupied) beds across ward types."""

    icu: int = Field(default=4, ge=0, description="ICU beds available")
    general: int = Field(default=20, ge=0, description="General ward beds available")
    observation: int = Field(default=10, ge=0, description="Observation bays available")

    @property
    def total(self) -> int:
        return self.icu + self.general + self.observation

    def is_full(self) -> bool:
        return self.total == 0


class StaffCount(BaseModel, frozen=True):
    """On-duty clinical staff."""

    doctors: int = Field(default=3, ge=0, description="On-duty doctors")
    nurses: int = Field(default=8, ge=0, description="On-duty nurses")


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------


class Patient(BaseModel, frozen=True):
    """
    Represents a single ED patient throughout their visit.

    Severity score loosely follows ESI (Emergency Severity Index):
      1 = critical, 5 = minor.

    ``assigned_tier`` tracks the current placement decision:
      - "Queue"      → waiting, no bed assigned yet
      - "ICU"        → assigned to ICU
      - "General"    → assigned to general ward
      - "Observation"→ assigned to observation bay
      - "Discharged" → released from ED
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = Field(..., description="Patient name")
    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    vitals: Vitals
    chief_complaint: str = Field(..., description="Presenting complaint")
    arrival_time: datetime = Field(default_factory=datetime.utcnow)
    severity_score: float = Field(
        ...,
        ge=1.0,
        le=5.0,
        description="Triage severity 1 (critical) – 5 (minor); may be fractional after deterioration",
    )
    assigned_tier: AssignedTier = Field(default="Queue")
    wait_time: int = Field(default=0, ge=0, description="Sim-steps spent in the queue")

    # -----------------------------------------------------------------------
    # Validators
    # -----------------------------------------------------------------------

    @field_validator("severity_score")
    @classmethod
    def clamp_severity(cls, v: float) -> float:
        """Keep severity within [1.0, 5.0] after any drift."""
        return round(max(1.0, min(5.0, v)), 2)

    # -----------------------------------------------------------------------
    # Derived helpers
    # -----------------------------------------------------------------------

    @property
    def ideal_tier(self) -> AssignedTier:
        """
        Rule-based ideal placement derived from severity:
          1-2 → ICU
          3   → General
          4-5 → Observation
        """
        if self.severity_score <= 2.0:
            return "ICU"
        elif self.severity_score <= 3.0:
            return "General"
        else:
            return "Observation"

    @property
    def is_stable(self) -> bool:
        """Patient considered stable if severity ≥ 3 (less urgent / minor)."""
        return self.severity_score >= 3.0

    def with_deterioration(self) -> "Patient":
        """Return a *new* Patient with severity worsened by 0.5."""
        return self.model_copy(
            update={
                "severity_score": min(5.0, self.severity_score + 0.5),
                "wait_time": self.wait_time + 1,
            }
        )

    def with_tier(self, tier: AssignedTier) -> "Patient":
        """Return a *new* Patient with the given tier assigned."""
        return self.model_copy(update={"assigned_tier": tier})

    def with_incremented_wait(self) -> "Patient":
        """Return a *new* Patient with wait_time incremented by 1."""
        return self.model_copy(update={"wait_time": self.wait_time + 1})


class EDState(BaseModel, frozen=True):
    """
    Complete snapshot of the Emergency Department at a single simulation step.

    ``waiting_patients``  – patients still in the triage queue (no bed).
    ``active_patients``   – patients currently occupying a bed / being treated.
    ``available_beds``    – inventory of free beds per ward type.
    ``staff``             – current on-duty staff counts.
    ``time_elapsed``      – wall-clock minutes simulated (each step ≈ 15 min).
    ``step_count``        – discrete environment step index.
    """

    waiting_patients: List[Patient] = Field(default_factory=list)
    active_patients: List[Patient] = Field(default_factory=list)
    available_beds: BedInventory = Field(default_factory=BedInventory)
    staff: StaffCount = Field(default_factory=StaffCount)
    time_elapsed: int = Field(default=0, ge=0, description="Simulated minutes elapsed")
    step_count: int = Field(default=0, ge=0)

    # -----------------------------------------------------------------------
    # Convenience properties
    # -----------------------------------------------------------------------

    @property
    def current_patient(self) -> Optional[Patient]:
        """Head-of-queue patient awaiting a triage decision, or None."""
        return self.waiting_patients[0] if self.waiting_patients else None

    @property
    def queue_length(self) -> int:
        return len(self.waiting_patients)

    @property
    def occupancy(self) -> int:
        return len(self.active_patients)

    def summary(self) -> str:
        """One-line human-readable state summary."""
        cp = self.current_patient
        cp_str = (
            f"{cp.name} (sev={cp.severity_score:.1f}, ideal={cp.ideal_tier})"
            if cp
            else "None"
        )
        return (
            f"[Step {self.step_count:>4d} | {self.time_elapsed:>5d}min] "
            f"Queue={self.queue_length} Active={self.occupancy} | "
            f"Beds(ICU={self.available_beds.icu} Gen={self.available_beds.general} "
            f"Obs={self.available_beds.observation}) | "
            f"Next: {cp_str}"
        )
