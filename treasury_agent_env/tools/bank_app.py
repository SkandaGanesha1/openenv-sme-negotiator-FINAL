"""Bank tool app — overdraft facility and covenant monitoring."""

from __future__ import annotations

from typing import Optional

from ..world_state import TreasuryWorldState


class BankApp:
    """
    Endpoints:
      - get_balances(sme_id)
      - draw_overdraft(sme_id, amount)
      - repay_overdraft(sme_id, amount)
      - view_covenants(sme_id)
    """

    def __init__(self, world: TreasuryWorldState) -> None:
        self._world = world

    def get_balances(self, sme_id: str) -> dict:
        cash = self._world.get_cash_balance(sme_id)
        od_drawn = self._world.get_overdraft_drawn(sme_id)
        od_limit = self._world.get_overdraft_limit(sme_id)
        od_rate = self._world.get_overdraft_rate(sme_id)
        od_interest = self._world._overdraft_interest.get(sme_id, 0.0)
        return {
            "app": "bank_app",
            "endpoint": "get_balances",
            "sme_id": sme_id,
            "cash_balance": round(cash, 2),
            "overdraft_drawn": round(od_drawn, 2),
            "overdraft_limit": round(od_limit, 2),
            "overdraft_headroom": round(od_limit - od_drawn, 2),
            "overdraft_utilization_pct": round(
                (od_drawn / od_limit * 100) if od_limit > 0 else 0.0, 2
            ),
            "accrued_overdraft_interest": round(od_interest, 2),
            "interest_rate_annual_pct": round(od_rate * 100, 2),
            "net_liquid_position": round(cash + (od_limit - od_drawn), 2),
        }

    def draw_overdraft(self, sme_id: str, amount: float) -> dict:
        result = self._world.apply_overdraft(sme_id, amount)
        result["app"] = "bank_app"
        result["endpoint"] = "draw_overdraft"
        if "error" not in result:
            result["message"] = (
                f"Drew INR {amount:,.2f} overdraft for {sme_id}. "
                f"Interest accrues at {self._world.get_overdraft_rate(sme_id)*100:.1f}% pa daily."
            )
        return result

    def repay_overdraft(self, sme_id: str, amount: float) -> dict:
        result = self._world.repay_overdraft(sme_id, amount)
        result["app"] = "bank_app"
        result["endpoint"] = "repay_overdraft"
        return result

    def view_covenants(self, sme_id: str) -> dict:
        od_limit = self._world.get_overdraft_limit(sme_id)
        od_drawn = self._world.get_overdraft_drawn(sme_id)
        od_rate = self._world.get_overdraft_rate(sme_id)
        utilization = (od_drawn / od_limit) if od_limit > 0 else 0.0

        # Covenant thresholds (realistic bank covenants)
        max_utilization = 0.80  # 80% utilization triggers review
        status = "OK"
        if utilization >= max_utilization:
            status = "COVENANT_BREACH_RISK"
        elif utilization >= 0.60:
            status = "ELEVATED"

        return {
            "app": "bank_app",
            "endpoint": "view_covenants",
            "sme_id": sme_id,
            "overdraft_limit": round(od_limit, 2),
            "current_drawn": round(od_drawn, 2),
            "utilization_pct": round(utilization * 100, 2),
            "max_allowed_utilization_pct": round(max_utilization * 100, 2),
            "interest_rate_annual_pct": round(od_rate * 100, 2),
            "covenant_status": status,
            "review_triggered_above_pct": 80.0,
            "note": (
                "Utilization above 80% may trigger bank review and facility reduction. "
                "Maintain below 60% for best terms at renewal."
            ),
        }
