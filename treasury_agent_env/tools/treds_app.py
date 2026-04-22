"""
TReDS tool app — Trade Receivables Discounting System simulation.

Mechanics modeled:
  - Reverse factoring: buyer creditworthiness determines the discount rate
  - Without-recourse: once discounted, SME has no liability if buyer defaults
  - Eligibility: Udyam-registered SMEs with GST-backed invoices
  - Rate determination: competitive bid simulation (buyer credit score → rate range)
"""

from __future__ import annotations

from random import Random
from typing import Optional

from ..world_state import TreasuryWorldState


class TredsApp:
    """
    Endpoints:
      - quote_discount_rate(invoice_id)
      - discount_invoice(invoice_id, min_accept_rate)
      - eligibility_summary(sme_id)
    """

    # Annual discount rate range (pa) based on buyer credit tiers
    _RATE_TABLE = [
        (0.85, 1.00, 0.06, 0.08),   # prime buyers: 6-8% pa
        (0.70, 0.85, 0.08, 0.11),   # good buyers: 8-11% pa
        (0.55, 0.70, 0.11, 0.13),   # average: 11-13% pa
        (0.00, 0.55, 0.13, 0.16),   # sub-prime: 13-16% pa
    ]

    def __init__(self, world: TreasuryWorldState, rng: Random) -> None:
        self._world = world
        self._rng = rng

    def _buyer_rate(self, buyer_id: str) -> tuple[float, float]:
        """Return (min_rate, max_rate) based on buyer credit score."""
        credit = self._world._graph.nodes.get(buyer_id, {}).get("credit_score", 0.70)
        for lo, hi, rlo, rhi in self._rate_TABLE if hasattr(self, "_rate_TABLE") else self._RATE_TABLE:
            if lo <= credit <= hi:
                return rlo, rhi
        return 0.10, 0.14

    def quote_discount_rate(self, invoice_id: str) -> dict:
        inv = self._world._invoices.get(invoice_id)
        if inv is None:
            return {"error": f"Invoice {invoice_id} not found", "app": "treds_app"}

        if inv["status"] != "pending":
            return {
                "error": f"Invoice {invoice_id} not eligible (status={inv['status']})",
                "app": "treds_app",
            }

        buyer_id = inv["buyer_id"]
        credit = self._world._graph.nodes.get(buyer_id, {}).get("credit_score", 0.70)

        # Find rate band
        rlo, rhi = 0.10, 0.14
        for lo, hi, band_lo, band_hi in self._RATE_TABLE:
            if lo <= credit <= hi:
                rlo, rhi = band_lo, band_hi
                break

        # Competitive bid: 3 fictitious financiers bid
        bids = [round(self._rng.uniform(rlo, rhi), 4) for _ in range(3)]
        best_rate = min(bids)

        face = inv["amount"]
        days_remaining = max(1, inv["due_day"] - self._world._current_day)
        discount_amount = face * best_rate * (days_remaining / 365.0)
        advance = face - discount_amount

        return {
            "app": "treds_app",
            "endpoint": "quote_discount_rate",
            "invoice_id": invoice_id,
            "buyer_id": buyer_id,
            "buyer_credit_score": round(credit, 3),
            "financier_bids": sorted(bids),
            "best_rate_annual_pct": round(best_rate * 100, 3),
            "face_value": round(face, 2),
            "days_remaining": days_remaining,
            "estimated_discount_amount": round(discount_amount, 2),
            "estimated_advance": round(advance, 2),
            "note": "Rate is without-recourse. SME has no liability on buyer default.",
        }

    def discount_invoice(
        self,
        invoice_id: str,
        min_accept_rate: Optional[float] = None,
    ) -> dict:
        """
        Execute TReDS discounting.
        If min_accept_rate is provided, only proceed if best bid ≤ min_accept_rate.
        """
        inv = self._world._invoices.get(invoice_id)
        if inv is None:
            return {"error": f"Invoice {invoice_id} not found", "app": "treds_app"}

        if inv["status"] != "pending":
            return {
                "error": f"Invoice {invoice_id} not eligible (status={inv['status']})",
                "app": "treds_app",
            }

        buyer_id = inv["buyer_id"]
        credit = self._world._graph.nodes.get(buyer_id, {}).get("credit_score", 0.70)

        rlo, rhi = 0.10, 0.14
        for lo, hi, band_lo, band_hi in self._RATE_TABLE:
            if lo <= credit <= hi:
                rlo, rhi = band_lo, band_hi
                break

        bids = [round(self._rng.uniform(rlo, rhi), 4) for _ in range(3)]
        best_rate = min(bids)

        if min_accept_rate is not None and best_rate > min_accept_rate:
            return {
                "app": "treds_app",
                "endpoint": "discount_invoice",
                "invoice_id": invoice_id,
                "accepted": False,
                "best_rate": round(best_rate * 100, 3),
                "min_accept_rate": round(min_accept_rate * 100, 3),
                "reason": "Best bid rate exceeds agent's minimum acceptance threshold.",
            }

        result = self._world.apply_treds_discount(invoice_id, best_rate)
        result["app"] = "treds_app"
        result["endpoint"] = "discount_invoice"
        result["accepted"] = True
        result["best_rate_annual_pct"] = round(best_rate * 100, 3)
        result["note"] = "Without-recourse: financier bears buyer default risk."
        return result

    def eligibility_summary(self, sme_id: str) -> dict:
        eligible_invoices = [
            inv for inv in self._world._invoices.values()
            if inv["sme_id"] == sme_id
            and inv["status"] == "pending"
            and not inv["treds_discounted"]
        ]
        total_eligible = sum(i["amount"] for i in eligible_invoices)
        # Check SME udyam registration from config
        sme_cfg = next(
            (s for s in self._world._cfg.smes if s.sme_id == sme_id), None
        )
        udyam = sme_cfg.udyam_registered if sme_cfg else True
        return {
            "app": "treds_app",
            "endpoint": "eligibility_summary",
            "sme_id": sme_id,
            "udyam_registered": udyam,
            "platform_registered": True,
            "eligible_invoice_count": len(eligible_invoices),
            "total_eligible_amount": round(total_eligible, 2),
            "treds_available": self._world._cfg.treds_available,
            "note": (
                "Mandatory for buyers with turnover > INR 250 Cr per RBI Nov 2024 notification. "
                "Eligible invoices can be discounted without recourse."
            ),
        }
