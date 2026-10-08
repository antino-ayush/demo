"""Reporting / Diffing: compares a run's average scores against a prior
baseline run and flags regressions per criterion -- this is what turns a
score into a CI gate instead of just a number.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CriterionDiff:
    criterion: str
    baseline: float | None
    current: float
    delta: float | None
    regressed: bool


@dataclass
class RunReport:
    run_id: str
    baseline_run_id: str | None
    diffs: list[CriterionDiff]

    @property
    def has_regression(self) -> bool:
        return any(d.regressed for d in self.diffs)

    def render(self) -> str:
        lines = [f"run {self.run_id} vs baseline {self.baseline_run_id or '(none)'}"]
        for d in self.diffs:
            marker = "FAIL" if d.regressed else "ok"
            baseline_str = f"{d.baseline:.3f}" if d.baseline is not None else "n/a"
            delta_str = f"{d.delta:+.3f}" if d.delta is not None else "n/a"
            lines.append(f"  [{marker}] {d.criterion}: {d.current:.3f} (baseline {baseline_str}, delta {delta_str})")
        return "\n".join(lines)


def diff_against_baseline(
    current_scores: dict[str, float],
    baseline_scores: dict[str, float] | None,
    run_id: str,
    baseline_run_id: str | None,
    regression_threshold: float = 0.02,
) -> RunReport:
    """Regression = current average score drops more than
    `regression_threshold` below the baseline run's average, for the same
    criterion."""
    diffs = []
    for criterion, current in sorted(current_scores.items()):
        baseline = (baseline_scores or {}).get(criterion)
        delta = None if baseline is None else current - baseline
        regressed = delta is not None and delta < -regression_threshold
        diffs.append(CriterionDiff(criterion, baseline, current, delta, regressed))
    return RunReport(run_id=run_id, baseline_run_id=baseline_run_id, diffs=diffs)
