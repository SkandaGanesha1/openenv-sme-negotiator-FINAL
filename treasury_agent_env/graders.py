"""Episode grading functions for TreasuryAgent."""

from __future__ import annotations

import math

from .models import TreasuryState
from .reward import MultiObjectiveRewardShaper, _strict_unit_interval

_shaper = MultiObjectiveRewardShaper()


def grade_treasury_balanced(state: TreasuryState) -> float:
    """
    Primary grader: balanced multi-objective scoring.

    Reads from state.world_snapshot (full hidden state) to compute terminal score.
    """
    snap = state.world_snapshot

    solvency_ok: bool = snap.get("solvency_ok", True)
    solvency_breach_day: int | None = snap.get("solvency_breach_day")
    total_financing_cost: float = snap.get("total_financing_cost", 0.0)
    total_revenue: float = sum(snap.get("revenue_collected", {}).values())
    vendor_overdue: dict = snap.get("vendor_overdue_days", {})
    max_overdue = max(vendor_overdue.values()) if vendor_overdue else 0.0
    vendor_stress_score = min(1.0, max_overdue / 45.0)

    # HHI — if not in snapshot, default to moderate concentration
    hhi: float = snap.get("hhi", 0.5)

    terminal, components = _shaper.terminal_reward(
        solvency_ok=solvency_ok,
        solvency_breach_day=solvency_breach_day,
        max_days=state.max_days,
        total_financing_cost=total_financing_cost,
        total_revenue=total_revenue,
        vendor_stress_score=vendor_stress_score,
        hhi=hhi,
    )
    return terminal


TASK_GRADERS: dict[str, object] = {
    "treasury_balanced": grade_treasury_balanced,
}
