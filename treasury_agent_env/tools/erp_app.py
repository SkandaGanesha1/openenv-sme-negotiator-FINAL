"""ERP tool app — invoice ledger queries and payment term updates."""

from __future__ import annotations

from typing import Optional

from ..world_state import TreasuryWorldState


class ErpApp:
    """
    Endpoints:
      - list_invoices(sme_id, buyer_id, status, window_days)
      - invoice_summary(sme_id, window_days)
      - projected_cashflow(sme_id, window_days)
      - update_terms(invoice_id, new_payment_days)
    """

    def __init__(self, world: TreasuryWorldState) -> None:
        self._world = world

    def list_invoices(
        self,
        sme_id: Optional[str] = None,
        buyer_id: Optional[str] = None,
        status: Optional[str] = None,
        window_days: Optional[int] = None,
    ) -> dict:
        invoices = self._world.get_invoices(
            sme_id=sme_id,
            buyer_id=buyer_id,
            status=status,
            window_days=window_days,
        )
        return {
            "app": "erp_app",
            "endpoint": "list_invoices",
            "count": len(invoices),
            "invoices": invoices[:50],  # cap at 50 to avoid context overflow
            "truncated": len(invoices) > 50,
        }

    def invoice_summary(
        self,
        sme_id: str,
        window_days: int = 30,
    ) -> dict:
        invoices = self._world.get_invoices(sme_id=sme_id, window_days=window_days)
        pending = [i for i in invoices if i["status"] == "pending"]
        overdue = [i for i in invoices if i["status"] == "overdue"]
        paid = [i for i in invoices if i["status"] == "paid"]
        discounted = [i for i in invoices if i["status"] == "discounted"]

        day = self._world._current_day

        def avg_days_outstanding(inv_list: list[dict]) -> float:
            if not inv_list:
                return 0.0
            return sum(day - i["issue_day"] for i in inv_list) / len(inv_list)

        total_receivables = sum(i["amount"] for i in pending + overdue)
        return {
            "app": "erp_app",
            "endpoint": "invoice_summary",
            "sme_id": sme_id,
            "window_days": window_days,
            "pending_count": len(pending),
            "overdue_count": len(overdue),
            "paid_count": len(paid),
            "discounted_count": len(discounted),
            "total_receivables": round(total_receivables, 2),
            "avg_days_outstanding": round(avg_days_outstanding(pending + overdue), 2),
            "largest_overdue_amount": round(
                max((i["amount"] for i in overdue), default=0.0), 2
            ),
        }

    def projected_cashflow(
        self,
        sme_id: str,
        window_days: int = 30,
    ) -> dict:
        projection = self._world.get_projected_cashflow(sme_id, window_days)
        projection["app"] = "erp_app"
        projection["endpoint"] = "projected_cashflow"
        return projection

    def update_terms(
        self,
        invoice_id: str,
        new_payment_days: Optional[int] = None,
    ) -> dict:
        if new_payment_days is None:
            return {"error": "new_payment_days is required", "app": "erp_app"}
        result = self._world.apply_term_update(invoice_id, new_payment_days)
        result["app"] = "erp_app"
        result["endpoint"] = "update_terms"
        return result
