"""
Compliance tool app — MSME 45-day payment rule (Section 43B(h)) enforcement.

Legal framework:
  - MSMED Act 2006: buyers must pay within 45 days of invoice acceptance
  - Section 43B(h) (effective Apr 1 2024): unpaid MSME dues not deductible
    for income tax in the year of expense, creating tax penalty for buyers
  - Samadhaan Portal: MSME complaint escalation → MSEFC state council arbitration
  - Side effect: filing Samadhaan increases buyer_power_score (damages relationship)
"""

from __future__ import annotations

from typing import Optional

from ..world_state import TreasuryWorldState


class ComplianceApp:
    """
    Endpoints:
      - check_45_day_breach(sme_id, buyer_id)
      - estimate_43B_tax_impact(buyer_id)
      - prepare_samadhaan_case(invoice_id)
    """

    _COMPOUND_RATE_MULTIPLIER = 3.0  # 3× RBI bank rate (approx 6% → 18% pa compounded)
    _RBI_BANK_RATE = 0.065           # 6.5% pa base

    def __init__(self, world: TreasuryWorldState) -> None:
        self._world = world

    def check_45_day_breach(
        self,
        sme_id: Optional[str] = None,
        buyer_id: Optional[str] = None,
    ) -> dict:
        """List all invoices where buyer has exceeded the 45-day payment window."""
        day = self._world._current_day
        breach_threshold = 45

        all_inv = self._world.get_invoices(sme_id=sme_id, buyer_id=buyer_id)
        breaches = []
        for inv in all_inv:
            if inv["status"] not in ("pending", "overdue"):
                continue
            acceptance_day = inv.get("acceptance_day", inv["issue_day"])
            days_since_acceptance = day - acceptance_day
            if days_since_acceptance > breach_threshold:
                penalty_interest = (
                    inv["amount"]
                    * self._COMPOUND_RATE_MULTIPLIER
                    * self._RBI_BANK_RATE
                    * (days_since_acceptance / 365.0)
                )
                breaches.append({
                    "invoice_id": inv["invoice_id"],
                    "buyer_id": inv["buyer_id"],
                    "sme_id": inv["sme_id"],
                    "amount": round(inv["amount"], 2),
                    "days_since_acceptance": days_since_acceptance,
                    "days_overdue_45d_rule": days_since_acceptance - breach_threshold,
                    "statutory_interest_owed": round(penalty_interest, 2),
                    "samadhaan_eligible": True,
                })

        total_overdue_amount = sum(b["amount"] for b in breaches)
        total_interest = sum(b["statutory_interest_owed"] for b in breaches)

        return {
            "app": "compliance_app",
            "endpoint": "check_45_day_breach",
            "sme_id": sme_id,
            "buyer_id": buyer_id,
            "breach_count": len(breaches),
            "total_overdue_amount": round(total_overdue_amount, 2),
            "total_statutory_interest_owed": round(total_interest, 2),
            "breaches": breaches[:20],
            "note": (
                "Buyers liable for compound interest at 3× RBI bank rate "
                "(currently ~19.5% pa) on delayed payments under MSMED Act 2006."
            ),
        }

    def estimate_43B_tax_impact(self, buyer_id: str) -> dict:
        """
        Estimate income tax disallowance for the buyer under Section 43B(h).
        Non-payment of MSME dues within 45 days makes the expense non-deductible
        in that financial year, creating a tax liability for the buyer.
        """
        buyer_data = self._world._graph.nodes.get(buyer_id, {})
        if not buyer_data:
            return {"error": f"Buyer {buyer_id} not found", "app": "compliance_app"}

        tax_rate = buyer_data.get("tax_rate", 0.25)
        day = self._world._current_day

        overdue_invoices = [
            inv for inv in self._world._invoices.values()
            if inv["buyer_id"] == buyer_id
            and inv["status"] in ("pending", "overdue")
            and (day - inv.get("acceptance_day", inv["issue_day"])) > 45
        ]

        total_disallowed = sum(i["amount"] for i in overdue_invoices)
        tax_impact = total_disallowed * tax_rate

        return {
            "app": "compliance_app",
            "endpoint": "estimate_43B_tax_impact",
            "buyer_id": buyer_id,
            "buyer_tax_rate_pct": round(tax_rate * 100, 1),
            "overdue_invoice_count": len(overdue_invoices),
            "total_disallowed_expense": round(total_disallowed, 2),
            "estimated_tax_disallowance": round(tax_impact, 2),
            "effective_from": "April 1, 2024",
            "note": (
                f"Buyer faces INR {tax_impact:,.0f} additional tax liability if dues remain unpaid. "
                "This creates negotiation leverage for the SME."
            ),
        }

    def prepare_samadhaan_case(self, invoice_id: str) -> dict:
        """
        File a complaint on the MSME Samadhaan portal.
        Side effect: buyer_power_score increases (relationship damaged).
        Benefit: increases buyer's probability of paying the specific invoice.
        """
        if not self._world._cfg.compliance_active:
            return {
                "error": (
                    "Compliance escalation not available for this task. "
                    "Samadhaan is enabled only for HARD difficulty."
                ),
                "app": "compliance_app",
            }

        inv = self._world._invoices.get(invoice_id)
        if inv is None:
            return {"error": f"Invoice {invoice_id} not found", "app": "compliance_app"}

        day = self._world._current_day
        acceptance_day = inv.get("acceptance_day", inv["issue_day"])
        days_since = day - acceptance_day

        if days_since <= 45:
            return {
                "error": (
                    f"Invoice {invoice_id} is only {days_since} days old. "
                    "Samadhaan requires 45+ days from acceptance."
                ),
                "app": "compliance_app",
            }

        result = self._world.flag_samadhaan(invoice_id)
        result["app"] = "compliance_app"
        result["endpoint"] = "prepare_samadhaan_case"
        result["days_since_acceptance"] = days_since
        result["warning"] = (
            "Filing Samadhaan increases buyer_power_score by 0.10, "
            "making future negotiations harder. Use only when other options exhausted."
        )
        result["expected_outcome"] = (
            "MSEFC will summon buyer within 60 days. Buyer now has elevated probability "
            "of paying this invoice to avoid formal arbitration."
        )
        return result
