"""
Rubric-based episode graders for TreasuryCommandCenter.

Grader hierarchy (lexicographic priority):
  1. Solvency (RLVR verifiable) — if breached, caps score ≤ 0.10
  2. Compliance (RLVR verifiable) — each breach deducts 0.05
  3. Financing efficiency — continuous cost ratio
  4. Vendor stress — continuous
  5. Concentration risk — HHI
  6. Tool usage quality — rubric average across episode
  7. World model accuracy — prediction MAE bonus
  8. Multi-agent coordination — coalition/oversight/manager bonuses

All graders return float in strict open interval (0, 1).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .reward import GRPORewardShaper, _strict_unit

_shaper = GRPORewardShaper()


def grade_tcc_solo(state_snapshot: Dict[str, Any]) -> float:
    """
    Solo Treasury Officer grader.
    Uses full multi-objective composite with RLVR override.
    """
    snap = state_snapshot
    solvency_ok: bool = snap.get("solvency_ok", True)
    solvency_breach_day: Optional[int] = snap.get("solvency_breach_day")
    max_days: int = snap.get("max_days", 180)
    total_cost: float = snap.get("total_financing_cost", 0.0)
    total_rev: float = sum(snap.get("revenue_collected", {}).values()) or 1.0
    vendor_stress: float = _vendor_stress_from_snap(snap)
    hhi: float = snap.get("hhi", 0.5)
    compliance_breaches: int = snap.get("compliance_breach_count", 0)
    tool_quality: float = snap.get("avg_tool_quality_score", 0.5)
    wm_error: float = snap.get("avg_world_model_error", 0.0)
    constitution_violated: bool = snap.get("constitution_violated", False)

    terminal, _ = _shaper.terminal_reward(
        solvency_ok=solvency_ok,
        solvency_breach_day=solvency_breach_day,
        max_days=max_days,
        total_financing_cost=total_cost,
        total_revenue=total_rev,
        vendor_stress_score=vendor_stress,
        hhi=hhi,
        tool_quality_score=tool_quality,
        world_model_error=wm_error,
        compliance_breach_count=compliance_breaches,
        constitution_violated=constitution_violated,
    )
    return terminal


def grade_tcc_multi(state_snapshot: Dict[str, Any]) -> float:
    """
    Multi-SME grader: average solo score across all SMEs with coordination bonus.
    CTDE bonus: reward is boosted when solvency is maintained across ALL SMEs.
    """
    per_sme: List[Dict[str, Any]] = state_snapshot.get("per_sme_snapshots", [])
    if not per_sme:
        return grade_tcc_solo(state_snapshot)

    scores = [grade_tcc_solo(s) for s in per_sme]
    avg_score = sum(scores) / len(scores)

    # CTDE coordination bonus: all solvent → +0.05
    all_solvent = all(s.get("solvency_ok", True) for s in per_sme)
    coordination_bonus = 0.05 if all_solvent else 0.0

    return _strict_unit(avg_score + coordination_bonus)


def grade_tcc_coalition(state_snapshot: Dict[str, Any]) -> float:
    """
    Coalition grader: multi-SME score + coalition coordination bonus.
    Bonus awarded when coalition channel was used AND financing cost improved.
    """
    base = grade_tcc_multi(state_snapshot)
    coalition_used = state_snapshot.get("coalition_messages_posted", 0) > 0
    financing_improved = state_snapshot.get("total_financing_cost", float("inf")) < state_snapshot.get("baseline_financing_cost", float("inf"))
    coalition_bonus = 0.05 if (coalition_used and financing_improved) else 0.0
    return _strict_unit(base + coalition_bonus)


def grade_tcc_oversight(state_snapshot: Dict[str, Any]) -> float:
    """
    Oversight grader: F1 on risk detection + intervention coverage.
    """
    gt_risky = state_snapshot.get("ground_truth_risky_smes", [])
    flagged = state_snapshot.get("total_flagged_smes", [])
    interventions = state_snapshot.get("interventions", {})

    from .reward import rubric_oversight_quality
    score = rubric_oversight_quality(flagged, gt_risky, interventions)
    return _strict_unit(score)


def grade_tcc_manager(state_snapshot: Dict[str, Any]) -> float:
    """
    Manager grader: world-level solvency + fairness + DSO improvement + cost.
    """
    total_smes = max(state_snapshot.get("total_smes", 1), 1)
    solvent_smes = state_snapshot.get("solvent_smes", total_smes)
    solvent_fraction = solvent_smes / total_smes

    gini = state_snapshot.get("gini_payment_days", 0.0)
    dso_improvement = state_snapshot.get("avg_dso_improvement_days", 0.0)
    total_cost = state_snapshot.get("total_financing_cost", 0.0)
    total_rev = sum(state_snapshot.get("revenue_collected", {}).values()) or 1.0
    instruction_quality = state_snapshot.get("avg_instruction_quality", 0.5)

    score, _ = _shaper.manager_reward(
        solvent_fraction=solvent_fraction,
        avg_dso_improvement=dso_improvement,
        gini_days=gini,
        total_financing_cost=total_cost,
        total_revenue=total_rev,
        instruction_quality=instruction_quality,
    )
    return score


def _vendor_stress_from_snap(snap: Dict[str, Any]) -> float:
    vendor_overdue = snap.get("vendor_overdue_days", {})
    if not vendor_overdue:
        return 0.0
    max_overdue = max(vendor_overdue.values()) if isinstance(vendor_overdue, dict) else float(vendor_overdue)
    return min(1.0, max_overdue / 45.0)


# ── Grader registry ────────────────────────────────────────────────────────────

TCC_TASK_GRADERS: Dict[str, Any] = {
    "tcc_solo":       grade_tcc_solo,
    "tcc_multi":      grade_tcc_multi,
    "tcc_coalition":  grade_tcc_coalition,
    "tcc_oversight":  grade_tcc_oversight,
    "tcc_manager":    grade_tcc_manager,
}

MODE_TO_GRADER: Dict[str, str] = {
    "treasury-solo":       "tcc_solo",
    "treasury-multi":      "tcc_multi",
    "treasury-coalition":  "tcc_coalition",
    "treasury-oversight":  "tcc_oversight",
    "treasury-manager":    "tcc_manager",
}
