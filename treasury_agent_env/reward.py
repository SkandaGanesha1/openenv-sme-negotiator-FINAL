"""
Multi-objective reward shaper for TreasuryAgent.

Four objectives with lexicographic priority and weighted scalarization:
  1. Solvency       (weight 0.50) — binary: all SMEs cash-positive each day
  2. Financing cost (weight 0.25) — minimize interest + discounting fees / revenue
  3. Vendor stress  (weight 0.15) — minimize days vendors are paid late
  4. Concentration  (weight 0.10) — minimize HHI buyer concentration

Solvency breach → override composite with -1.0 (catastrophic signal).
Final reward is mapped to strict open interval (0, 1) for grader compatibility.
"""

from __future__ import annotations

import math


_EPS = 1e-6
_W_SOLVENCY = 0.50
_W_COST = 0.25
_W_VENDOR = 0.15
_W_CONCENTRATION = 0.10


def _unit_clamp(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _strict_unit_interval(score: float) -> float:
    """Map scores into strict open interval (0, 1) — matches existing grader convention."""
    value = float(score)
    if not math.isfinite(value):
        return _EPS
    return float(min(1.0 - _EPS, max(_EPS, value)))


class MultiObjectiveRewardShaper:
    """
    Computes step and terminal rewards from world-state metrics.

    Step reward:  partial signal based on current KPI snapshot.
    Terminal reward: full episode scoring used by the grader.
    """

    def __init__(
        self,
        w_solvency: float = _W_SOLVENCY,
        w_cost: float = _W_COST,
        w_vendor: float = _W_VENDOR,
        w_concentration: float = _W_CONCENTRATION,
    ) -> None:
        assert abs(w_solvency + w_cost + w_vendor + w_concentration - 1.0) < 1e-6, (
            "Reward weights must sum to 1.0"
        )
        self._w = {
            "solvency": w_solvency,
            "cost": w_cost,
            "vendor": w_vendor,
            "concentration": w_concentration,
        }

    # ── Objective components ─────────────────────────────────────────────────

    def solvency_component(self, solvency_ok: bool) -> float:
        return 1.0 if solvency_ok else 0.0

    def financing_cost_component(
        self, total_financing_cost: float, total_revenue: float
    ) -> float:
        if total_revenue <= 0:
            return 0.5  # neutral when no revenue yet
        cost_ratio = total_financing_cost / total_revenue
        # cost_ratio of 0 → 1.0; cost_ratio of 0.10 (10%) → 0.0
        return _unit_clamp(1.0 - cost_ratio / 0.10)

    def vendor_stress_component(self, vendor_stress_score: float) -> float:
        return _unit_clamp(1.0 - vendor_stress_score)

    def concentration_component(self, hhi: float) -> float:
        # HHI of 0 (perfectly diverse) → 1.0; HHI of 1.0 (monopoly) → 0.0
        return _unit_clamp(1.0 - hhi)

    # ── Composite reward ─────────────────────────────────────────────────────

    def compute(
        self,
        *,
        solvency_ok: bool,
        total_financing_cost: float,
        total_revenue: float,
        vendor_stress_score: float,
        hhi: float,
    ) -> tuple[float, dict]:
        """
        Returns (raw_composite, component_dict).
        Solvency breach overrides composite to near-zero.
        """
        s = self.solvency_component(solvency_ok)
        c = self.financing_cost_component(total_financing_cost, total_revenue)
        v = self.vendor_stress_component(vendor_stress_score)
        k = self.concentration_component(hhi)

        composite = (
            self._w["solvency"] * s
            + self._w["cost"] * c
            + self._w["vendor"] * v
            + self._w["concentration"] * k
        )

        if not solvency_ok:
            # Catastrophic override: solvency breach dominates (strictly below 0.10)
            composite = min(composite, 0.09)

        return composite, {
            "solvency": round(s, 4),
            "financing_cost": round(c, 4),
            "vendor_stress": round(v, 4),
            "concentration": round(k, 4),
            "composite": round(composite, 4),
        }

    def step_reward(
        self,
        *,
        solvency_ok: bool,
        total_financing_cost: float,
        total_revenue: float,
        vendor_stress_score: float,
        hhi: float,
        tool_was_useful: bool = True,
    ) -> tuple[float, dict]:
        """
        Per-step partial reward.  Scaled to [0, 0.3] so terminal reward
        dominates the episode signal.
        """
        composite, components = self.compute(
            solvency_ok=solvency_ok,
            total_financing_cost=total_financing_cost,
            total_revenue=total_revenue,
            vendor_stress_score=vendor_stress_score,
            hhi=hhi,
        )
        # Idle tool call (observe-only) still gets partial credit for not breaking anything
        scale = 0.3 if tool_was_useful else 0.15
        partial = _strict_unit_interval(composite * scale)
        return partial, components

    def terminal_reward(
        self,
        *,
        solvency_ok: bool,
        solvency_breach_day: int | None,
        max_days: int,
        total_financing_cost: float,
        total_revenue: float,
        vendor_stress_score: float,
        hhi: float,
    ) -> tuple[float, dict]:
        """
        Full episode terminal reward.  Caps at 0.99 to avoid exact endpoint.
        Solvency breaches are penalized proportionally to how early they occurred.
        """
        composite, components = self.compute(
            solvency_ok=solvency_ok,
            total_financing_cost=total_financing_cost,
            total_revenue=total_revenue,
            vendor_stress_score=vendor_stress_score,
            hhi=hhi,
        )
        if not solvency_ok and solvency_breach_day is not None:
            # Earlier breach = harsher penalty
            fraction_survived = solvency_breach_day / max(1, max_days)
            composite = composite * fraction_survived * 0.5

        terminal = _strict_unit_interval(min(composite, 0.99))
        components["terminal"] = terminal
        return terminal, components
