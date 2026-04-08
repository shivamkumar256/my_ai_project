"""
grader.py
=========
AgentGrader – normalised scoring for completed ED Triage episodes.

Usage
-----
>>> from ed_triage.grader import AgentGrader
>>> from ed_triage.tasks import TASK_MEDIUM
>>> grader = AgentGrader()
>>> score, breakdown = grader.grade(env.episode_log, task="medium")
>>> print(f"Final score: {score:.3f}")
>>> print(breakdown)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .tasks import TASK_REGISTRY, TaskConfig


# ---------------------------------------------------------------------------
# ScoreBreakdown  (returned alongside the scalar grade)
# ---------------------------------------------------------------------------


@dataclass
class ScoreBreakdown:
    """
    Per-dimension scoring detail for a single graded episode.

    Attributes
    ----------
    task : str
        Name of the task that was graded.
    total : float
        Weighted sum of dimension scores in [0.0, 1.0].
    dimensions : dict[str, float]
        Raw (unweighted) score in [0.0, 1.0] for each grading dimension.
    weights : dict[str, float]
        Weight applied to each dimension.
    weighted : dict[str, float]
        ``dimensions[d] * weights[d]`` for each dimension ``d``.
    passed : bool
        ``True`` if *all* success criteria were met.
    criteria_results : dict[str, bool]
        Per-criterion pass/fail flags.
    notes : list[str]
        Human-readable explanations for each scoring decision.
    """

    task: str = ""
    total: float = 0.0
    dimensions: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    weighted: Dict[str, float] = field(default_factory=dict)
    passed: bool = False
    criteria_results: Dict[str, bool] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"[AgentGrader] Task={self.task!r}  score={self.total:.3f}  "
            f"{'PASS' if self.passed else 'FAIL'}",
            "  Dimensions:",
        ]
        for dim, score in self.dimensions.items():
            w = self.weights.get(dim, 0.0)
            lines.append(
                f"    {dim:<22} raw={score:.3f}  weight={w:.2f}  "
                f"contrib={self.weighted.get(dim, 0.0):.3f}"
            )
        if self.criteria_results:
            lines.append("  Criteria:")
            for crit, ok in self.criteria_results.items():
                lines.append(f"    {crit:<30} {'OK' if ok else 'FAIL'}")
        if self.notes:
            lines.append("  Notes:")
            for note in self.notes:
                lines.append(f"    - {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# AgentGrader
# ---------------------------------------------------------------------------


class AgentGrader:
    """
    Computes a normalised score in ``[0.0, 1.0]`` for a completed episode.

    Each call to :meth:`grade` analyses the episode log produced by
    ``EDTriageEnv.episode_log`` and returns:

    * a scalar ``float`` grade, and
    * a :class:`ScoreBreakdown` with per-dimension detail.

    The grader is stateless – the same instance can be reused across episodes
    and tasks.

    Parameters
    ----------
    strict : bool
        If ``True``, any failed mandatory criterion sets the total score to
        0.0 regardless of dimension scores.  Default ``False``.
    """

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def grade(
        self,
        episode_log: List[Dict[str, Any]],
        task: str,
    ) -> Tuple[float, ScoreBreakdown]:
        """
        Grade a completed episode against the named task.

        Parameters
        ----------
        episode_log : list[dict]
            Per-step log as returned by ``EDTriageEnv.episode_log``.
            Each entry must contain at minimum:
              * ``"action"``           – ``Action`` enum value
              * ``"reward"``           – float (scalar reward for this step)
              * ``"patient"``          – ``Patient`` or ``None``
              * ``"reward_breakdown"`` – ``RewardBreakdown``
              * ``"done"``             – bool (True on last step)

            .. note::
               ``EDTriageEnv.episode_log`` does not include a ``"reward"``
               key directly – it is derived from ``reward_breakdown.total``
               inside this method, so you do **not** need to add it manually.

        task : str
            One of ``"easy"``, ``"medium"``, ``"hard"`` (case-insensitive).

        Returns
        -------
        score : float
            Normalised score in ``[0.0, 1.0]``.
        breakdown : ScoreBreakdown
            Detailed per-dimension results.
        """
        cfg = TASK_REGISTRY[task.strip().lower()]
        breakdown = ScoreBreakdown(task=cfg.name, weights=dict(cfg.score_weights))

        # Dispatch to the appropriate grading strategy
        if cfg.name == "easy":
            self._grade_easy(episode_log, cfg, breakdown)
        elif cfg.name == "medium":
            self._grade_medium(episode_log, cfg, breakdown)
        elif cfg.name == "hard":
            self._grade_hard(episode_log, cfg, breakdown)
        else:
            raise ValueError(f"No grading strategy for task '{cfg.name}'")

        # Compute weighted total
        total = sum(
            breakdown.dimensions.get(dim, 0.0) * w
            for dim, w in cfg.score_weights.items()
        )
        breakdown.weighted = {
            dim: breakdown.dimensions.get(dim, 0.0) * w
            for dim, w in cfg.score_weights.items()
        }

        # Strict mode: any failed criterion zeroes the entire score
        all_passed = all(breakdown.criteria_results.values())
        if self.strict and not all_passed:
            total = 0.0
            breakdown.notes.append(
                "Score zeroed: strict mode + one or more criteria failed."
            )

        breakdown.total = round(max(0.0, min(1.0, total)), 4)
        breakdown.passed = all_passed
        return breakdown.total, breakdown

    # ------------------------------------------------------------------
    # Task-specific grading strategies
    # ------------------------------------------------------------------

    def _grade_easy(
        self,
        log: List[Dict[str, Any]],
        cfg: TaskConfig,
        bd: ScoreBreakdown,
    ) -> None:
        """
        EASY grading: accuracy = fraction of assignment actions that match ideal tier.

        Scoring dimension: ``"accuracy"``
        Success criterion: ``tier_accuracy >= 1.0`` (100%)
        """
        from .actions import Action

        assignment_steps = [
            e for e in log
            if e["patient"] is not None
            and e["action"] not in (Action.HOLD_QUEUE,)
        ]
        if not assignment_steps:
            bd.dimensions["accuracy"] = 0.0
            bd.notes.append("No assignment actions found.")
            bd.criteria_results["tier_accuracy"] = False
            return

        correct = sum(
            1
            for e in assignment_steps
            if (
                e["action"] != Action.DISCHARGE
                and e["action"].tier_name == e["patient"].ideal_tier
            )
        )
        accuracy = correct / len(assignment_steps)
        bd.dimensions["accuracy"] = accuracy
        bd.notes.append(
            f"Correct placements: {correct}/{len(assignment_steps)} = {accuracy:.1%}"
        )

        threshold = cfg.success_criteria.get("tier_accuracy", 1.0)
        passed = accuracy >= threshold
        bd.criteria_results["tier_accuracy"] = passed
        if not passed:
            bd.notes.append(
                f"Criterion FAIL: tier_accuracy {accuracy:.1%} < {threshold:.1%}"
            )

    def _grade_medium(
        self,
        log: List[Dict[str, Any]],
        cfg: TaskConfig,
        bd: ScoreBreakdown,
    ) -> None:
        """
        MEDIUM grading:

        Dimensions
        ----------
        throughput (60%) – fraction of patients assigned within avg_wait threshold
        critical_safety (40%) – 1.0 if zero critical misses, else 0.0

        Success criteria
        ----------------
        * zero_critical_misses
        * avg_wait_steps < max_avg_wait_steps
        """
        from .actions import Action

        # -- Critical misses --------------------------------------------------
        n_critical_misses = sum(
            1
            for e in log
            if (
                e["patient"] is not None
                and e["patient"].severity_score <= 2.0
                and e["action"] in (Action.ASSIGN_OBSERVATION, Action.HOLD_QUEUE)
            )
        )
        zero_misses = n_critical_misses == 0
        bd.criteria_results["zero_critical_misses"] = zero_misses
        bd.dimensions["critical_safety"] = 1.0 if zero_misses else max(
            0.0, 1.0 - (n_critical_misses * 0.25)
        )
        bd.notes.append(f"Critical misses: {n_critical_misses}")

        # -- Throughput / avg wait -------------------------------------------
        wait_times = [
            e["patient"].wait_time
            for e in log
            if e["patient"] is not None
            and e["action"] not in (Action.HOLD_QUEUE,)
        ]
        avg_wait = sum(wait_times) / max(len(wait_times), 1)
        max_wait = cfg.success_criteria.get("max_avg_wait_steps", 6.0)

        # Score is 1.0 at avg_wait=0, linearly declining to 0.0 at 2*max_wait
        throughput_score = max(0.0, 1.0 - avg_wait / (2.0 * max_wait))
        bd.dimensions["throughput"] = throughput_score
        bd.criteria_results["max_avg_wait_steps"] = avg_wait <= max_wait
        bd.notes.append(
            f"Avg wait: {avg_wait:.2f} steps (threshold <= {max_wait})"
        )

    def _grade_hard(
        self,
        log: List[Dict[str, Any]],
        cfg: TaskConfig,
        bd: ScoreBreakdown,
    ) -> None:
        """
        HARD grading:

        Dimensions
        ----------
        mortality (50%)    – fraction of critical patients who were NOT missed
        utilization (30%)  – proportion of steps where at least one bed was active
        speed (20%)        – inverse of mean wait time (capped)

        Success criteria
        ----------------
        * mortality_rate < max_mortality_rate
        * resource utilization > min_utilization
        """
        from .actions import Action

        # -- Mortality -------------------------------------------------------
        # Proxy: a critical patient (sev <= 2) sent to Observation or held = "death"
        critical_actions = [
            e for e in log
            if e["patient"] is not None and e["patient"].severity_score <= 2.0
        ]
        missed_critical = sum(
            1 for e in critical_actions
            if e["action"] in (Action.ASSIGN_OBSERVATION, Action.HOLD_QUEUE)
        )
        n_critical = max(len(critical_actions), 1)
        mortality_rate = missed_critical / n_critical
        max_mortality = cfg.success_criteria.get("max_mortality_rate", 0.05)

        # Score: 1.0 at zero mortality, drops linearly
        mortality_score = max(0.0, 1.0 - mortality_rate / max(max_mortality * 2, 0.01))
        bd.dimensions["mortality"] = mortality_score
        bd.criteria_results["max_mortality_rate"] = mortality_rate < max_mortality
        bd.notes.append(
            f"Mortality (proxy): {missed_critical}/{n_critical} = {mortality_rate:.1%} "
            f"(threshold < {max_mortality:.1%})"
        )

        # -- Utilization -----------------------------------------------------
        # Fraction of steps where at least one bed was occupied (active patients > 0)
        steps_with_active = sum(
            1 for e in log
            if e.get("state_snapshot") is not None
            and len(e["state_snapshot"].active_patients) > 0
        )
        # Fallback: derive from reward_breakdown bed overflow signals
        if steps_with_active == 0:
            steps_with_active = sum(
                1 for e in log
                if e["reward_breakdown"].overflow_penalty < 0
                or e["reward_breakdown"].tier_reward > 0
            )
        n_steps = max(len(log), 1)
        utilization = steps_with_active / n_steps
        min_util = cfg.success_criteria.get("min_utilization", 0.75)
        utilization_score = min(1.0, utilization / min_util)
        bd.dimensions["utilization"] = utilization_score
        bd.criteria_results["min_utilization"] = utilization >= min_util
        bd.notes.append(
            f"Utilization (proxy): {utilization:.1%} "
            f"(threshold > {min_util:.1%})"
        )

        # -- Speed -----------------------------------------------------------
        wait_times = [
            e["patient"].wait_time
            for e in log
            if e["patient"] is not None
            and e["action"] not in (Action.HOLD_QUEUE,)
        ]
        avg_wait = sum(wait_times) / max(len(wait_times), 1)
        # Score: 1.0 at avg_wait <= 2, decays to 0 at avg_wait >= 12
        speed_score = max(0.0, min(1.0, 1.0 - (avg_wait - 2.0) / 10.0))
        bd.dimensions["speed"] = speed_score
        bd.notes.append(f"Avg wait: {avg_wait:.2f} steps")

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _total_patients_acted_on(log: List[Dict[str, Any]]) -> int:
        """Count steps where an action was taken on a real patient."""
        from .actions import Action
        return sum(1 for e in log if e["patient"] is not None)

    @staticmethod
    def _cumulative_reward(log: List[Dict[str, Any]]) -> float:
        """Sum of all step rewards derived from reward_breakdown.total."""
        return sum(e["reward_breakdown"].total for e in log)

    def summary(self, episode_log: List[Dict[str, Any]], task: str) -> str:
        """
        One-line summary string for quick logging.

        Returns
        -------
        str
            E.g. ``"[easy] score=0.933 PASS  accuracy=0.933"``
        """
        score, bd = self.grade(episode_log, task)
        dim_str = "  ".join(
            f"{k}={v:.3f}" for k, v in bd.dimensions.items()
        )
        return f"[{task}] score={score:.3f} {'PASS' if bd.passed else 'FAIL'}  {dim_str}"
