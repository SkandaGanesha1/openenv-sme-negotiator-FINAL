"""
Dynamic Discounting tool app.

Buyer-funded early payment: SME offers a discount in exchange for
early payment.  Unlike TReDS (bank-funded, without recourse), DD uses
the buyer's own surplus cash.  No bank involved; discount is mutual.

Key formula: annualised_discount = discount_pct / (payment_terms_days - target_days) * 365
"""

from __future__ import annotations

from random import Random
from typing import Optional

from ..world_state import TreasuryWorldState


class DdApp:
    """
    Endpoints:
      - propose_discount_scheme(buyer_id, target_days, max_discount_pct)
      - simulate_scheme(buyer_id, params)
      - activate_scheme(buyer_id, params)
    """

    def __init__(self, world: TreasuryWorldState, rng: Random) -> None:
        self._world = world
        self._rng = rng

    def _acceptance_probability(
        self, buyer_id: str, target_days: int, discount_pct: float
    ) -> float:
        """
        Buyers with excess liquidity and low power scores accept more readily.
        Higher discount_pct → higher acceptance.
        Shorter target_days (earlier payment) → lower acceptance (more cash required).
        """
        buyer_data = self._world._graph.nodes.get(buyer_id, {})
        buyer_power = buyer_data.get("buyer_power", 0.5)
        current_days = buyer_data.get("payment_days", 60)

        days_acceleration = max(0, current_days - target_days)
        # Annualised equivalent yield for buyer
        if days_acceleration > 0:
            annualised_yield = discount_pct / days_acceleration * 365
        else:
            annualised_yield = 0.0

        # Buyer accepts if yield > their hurdle rate (~8-12% pa for large corporates)
        hurdle = 0.08 + buyer_power * 0.04  # high-power buyers demand more
        attractiveness = min(1.0, annualised_yield / hurdle)
        # Modulated by buyer power (powerful buyers are harder to persuade)
        accept_prob = attractiveness * (1.0 - 0.3 * buyer_power)
        return max(0.0, min(0.90, accept_prob))

    def propose_discount_scheme(
        self,
        buyer_id: str,
        target_days: int,
        max_discount_pct: float,
    ) -> dict:
        buyer_data = self._world._graph.nodes.get(buyer_id, {})
        if not buyer_data:
            return {"error": f"Buyer {buyer_id} not found", "app": "dd_app"}

        current_days = buyer_data.get("payment_days", 60)
        accept_prob = self._acceptance_probability(buyer_id, target_days, max_discount_pct)

        # Pending receivables from this buyer
        pending = [
            inv for inv in self._world._invoices.values()
            if inv["buyer_id"] == buyer_id and inv["status"] == "pending"
        ]
        total_pending = sum(i["amount"] for i in pending)
        days_acceleration = max(0, current_days - target_days)
        cost_if_accepted = total_pending * max_discount_pct if days_acceleration > 0 else 0.0

        return {
            "app": "dd_app",
            "endpoint": "propose_discount_scheme",
            "buyer_id": buyer_id,
            "current_payment_days": current_days,
            "target_days": target_days,
            "max_discount_pct": round(max_discount_pct * 100, 3),
            "days_acceleration": days_acceleration,
            "estimated_acceptance_probability": round(accept_prob, 3),
            "total_pending_receivables": round(total_pending, 2),
            "estimated_discount_cost_if_accepted": round(cost_if_accepted, 2),
            "note": (
                "Proposal is not yet committed. Use activate_scheme to proceed. "
                "Use simulate_scheme to model cashflow impact first."
            ),
        }

    def simulate_scheme(self, buyer_id: str, params: dict) -> dict:
        """Cashflow impact projection without committing to the scheme."""
        target_days = params.get("target_days", 45)
        max_discount_pct = params.get("max_discount_pct", 0.02)

        buyer_data = self._world._graph.nodes.get(buyer_id, {})
        current_days = buyer_data.get("payment_days", 60)
        days_gain = max(0, current_days - target_days)

        pending = [
            inv for inv in self._world._invoices.values()
            if inv["buyer_id"] == buyer_id and inv["status"] == "pending"
        ]
        total = sum(i["amount"] for i in pending)

        # Scenario: base, stress (buyer rejects), optimistic (buyer partial acceptance)
        base_advance = total * (1.0 - max_discount_pct)
        optimistic_advance = total * 0.60 * (1.0 - max_discount_pct * 0.8)
        stress_advance = 0.0  # buyer rejects

        # Alternative: what would TReDS cost on same amount?
        credit = buyer_data.get("credit_score", 0.70)
        treds_rate_approx = 0.08 + (1.0 - credit) * 0.08  # rough estimate
        treds_cost = total * treds_rate_approx * (current_days / 365.0)

        return {
            "app": "dd_app",
            "endpoint": "simulate_scheme",
            "buyer_id": buyer_id,
            "params": params,
            "scenarios": {
                "base_case_advance": round(base_advance, 2),
                "optimistic_advance": round(optimistic_advance, 2),
                "stress_advance": round(stress_advance, 2),
                "dd_discount_cost": round(total * max_discount_pct, 2),
                "vs_treds_cost": round(treds_cost, 2),
                "dd_saving_vs_treds": round(treds_cost - total * max_discount_pct, 2),
            },
            "days_gained": days_gain,
            "recommendation": (
                "DD preferred over TReDS" if treds_cost > total * max_discount_pct * 1.2
                else "TReDS may be cheaper; compare rates"
            ),
        }

    def activate_scheme(self, buyer_id: str, params: dict) -> dict:
        """Commit to the DD scheme — mutates buyer payment behavior."""
        if not self._world._cfg.dd_available:
            return {
                "error": "Dynamic discounting not available for this task configuration.",
                "app": "dd_app",
            }
        target_days = params.get("target_days", 45)
        max_discount_pct = params.get("max_discount_pct", 0.02)

        result = self._world.activate_dd_scheme(buyer_id, {
            "target_days": target_days,
            "max_discount_pct": max_discount_pct,
        })
        result["app"] = "dd_app"
        result["endpoint"] = "activate_scheme"
        return result
