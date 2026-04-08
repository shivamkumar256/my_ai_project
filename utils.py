"""
utils.py
========
Utility helpers: patient name lists, random patient generation, and
episode-scenario factories.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import List, Optional

from .schemas import BedInventory, Patient, StaffCount, Vitals

# ---------------------------------------------------------------------------
# Realistic name pool
# ---------------------------------------------------------------------------
_FIRST_NAMES = [
    "Aarav", "Aisha", "Carlos", "Diana", "Ethan", "Fatima", "George",
    "Hannah", "Ivan", "Julia", "Kevin", "Leyla", "Marcus", "Nadia",
    "Omar", "Priya", "Quinn", "Rosa", "Samuel", "Tanya", "Umar",
    "Valentina", "Wei", "Xena", "Yusuf", "Zara",
]
_LAST_NAMES = [
    "Ahmed", "Brown", "Chen", "Davis", "Evans", "Fernandez", "Garcia",
    "Hassan", "Ibrahim", "Jones", "Kim", "Lee", "Martinez", "Nguyen",
    "Okafor", "Patel", "Rodriguez", "Singh", "Taylor", "Uddin",
    "Vargas", "Williams", "Xu", "Yang", "Zaidi",
]

# Chief complaints keyed by severity tier
_COMPLAINTS: dict[str, List[str]] = {
    "critical": [
        "cardiac arrest", "massive haemorrhage", "severe respiratory failure",
        "acute stroke", "septic shock",
    ],
    "emergent": [
        "chest pain with diaphoresis", "altered mental status",
        "severe asthma exacerbation", "high-grade fever + seizure",
    ],
    "urgent": [
        "moderate dyspnea", "abdominal pain", "head injury",
        "hyperglycaemia", "lacerations requiring repair",
    ],
    "less_urgent": [
        "sprained ankle", "ear pain", "mild vomiting", "UTI symptoms",
        "back pain", "rash",
    ],
    "non_urgent": [
        "medication refill request", "minor abrasion", "sore throat",
        "cold symptoms", "mild headache",
    ],
}

_SEVERITY_TO_COMPLAINT = {
    1: "critical",
    2: "emergent",
    3: "urgent",
    4: "less_urgent",
    5: "non_urgent",
}


# ---------------------------------------------------------------------------
# Vital sign generators per severity level
# ---------------------------------------------------------------------------

def _generate_vitals(severity: int, rng: random.Random) -> Vitals:
    """
    Generate physiologically plausible vitals that correspond to *severity*.

    severity 1 = critical → abnormal vitals
    severity 5 = non-urgent → normal vitals
    """
    if severity == 1:
        # Critical: BP crash or hypertensive crisis, tachycardia, low SpO2, fever
        return Vitals(
            bp_systolic=rng.randint(60, 85),
            bp_diastolic=rng.randint(40, 55),
            hr=rng.randint(130, 200),
            spo2=round(rng.uniform(72.0, 85.0), 1),
            temp=round(rng.uniform(38.8, 40.5), 1),
        )
    elif severity == 2:
        return Vitals(
            bp_systolic=rng.randint(85, 105),
            bp_diastolic=rng.randint(55, 70),
            hr=rng.randint(110, 135),
            spo2=round(rng.uniform(85.0, 92.0), 1),
            temp=round(rng.uniform(38.2, 39.5), 1),
        )
    elif severity == 3:
        return Vitals(
            bp_systolic=rng.randint(105, 145),
            bp_diastolic=rng.randint(65, 90),
            hr=rng.randint(88, 115),
            spo2=round(rng.uniform(92.0, 96.0), 1),
            temp=round(rng.uniform(37.2, 38.5), 1),
        )
    elif severity == 4:
        return Vitals(
            bp_systolic=rng.randint(115, 140),
            bp_diastolic=rng.randint(70, 85),
            hr=rng.randint(72, 95),
            spo2=round(rng.uniform(95.0, 98.5), 1),
            temp=round(rng.uniform(36.5, 37.5), 1),
        )
    else:  # severity == 5
        return Vitals(
            bp_systolic=rng.randint(115, 130),
            bp_diastolic=rng.randint(72, 82),
            hr=rng.randint(65, 85),
            spo2=round(rng.uniform(97.0, 99.5), 1),
            temp=round(rng.uniform(36.4, 37.2), 1),
        )


def generate_patient(
    severity: Optional[int] = None,
    rng: Optional[random.Random] = None,
    arrival_time: Optional[datetime] = None,
) -> Patient:
    """
    Generate a random ``Patient`` with clinically coherent vitals and complaint.

    Parameters
    ----------
    severity:
        If provided (1-5), generates a patient of that exact severity.
        Otherwise, severity is drawn from a realistic weighted distribution.
    rng:
        A ``random.Random`` instance for reproducible generation.
        Defaults to the module-level RNG.
    arrival_time:
        Arrival timestamp.  Defaults to UTC now.

    Returns
    -------
    Patient
    """
    rng = rng or random.Random()
    if severity is None:
        # Realistic ED distribution: majority moderate/minor
        severity = rng.choices(
            population=[1, 2, 3, 4, 5],
            weights=[5, 15, 35, 30, 15],
            k=1,
        )[0]

    name = f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"
    age = rng.randint(0, 95)
    vitals = _generate_vitals(severity, rng)
    complaint_key = _SEVERITY_TO_COMPLAINT[severity]
    complaint = rng.choice(_COMPLAINTS[complaint_key])

    return Patient(
        name=name,
        age=age,
        vitals=vitals,
        chief_complaint=complaint,
        arrival_time=arrival_time or datetime.now(timezone.utc),
        severity_score=float(severity),
    )


def generate_arrival_wave(
    n: int = 10,
    seed: Optional[int] = None,
) -> List[Patient]:
    """
    Generate a list of *n* patients simulating an ED arrival wave.

    Parameters
    ----------
    n:
        Number of patients to generate.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    List[Patient]
        Ordered by arrival (first in list = first in queue).
    """
    rng = random.Random(seed)
    now = datetime.now(timezone.utc)
    return [generate_patient(rng=rng, arrival_time=now) for _ in range(n)]


def default_beds() -> BedInventory:
    """Default ED bed inventory for a mid-sized hospital."""
    return BedInventory(icu=4, general=20, observation=10)


def default_staff() -> StaffCount:
    """Default on-duty staffing for a standard shift."""
    return StaffCount(doctors=3, nurses=8)
