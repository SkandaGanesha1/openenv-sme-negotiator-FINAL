"""
Analytics tool app — portfolio risk analysis, scenario simulation, KPI dashboard.

This is the agent's primary "observe and plan" tool.
The environment calls these before taking expensive financial actions.
"""

from __future__ import annotations

from typing import Optional

from ..world_state import TreasuryWorldState
from ..belief_state import TreasuryBeliefState


class AnalyticsApp:
    """
    Endpoints:
      - portfolio_risks()
      - scenario_analysis(sme_id, buyer_id)
      - kpi_dashboard(sme_id)
    """

    def __init__(self, world: TreasuryWorldState, belief: TreasuryBeliefState) -> None:
        self._world = world
        self._belief = belief

    def portfolio_risks(self) -> dict:
        """
        Compute HHI concentration, top risks, and buyer reliability.
        Uses belief state so agent sees uncertainty estimates, not raw credit scores.
        """
        cfg = self._world._cfg
        day = self._world._current_day

        # Revenue concentration HHI
        buyer_revenue: dict[str, float] = {}
        for sme in cfg.smes:
            for buyer in cfg.buyers:
                if self._world._graph.has_edge(sme.sme_id, buyer.buyer_id):
                    vol = self._world._graph[sme.sme_id][buyer.buyer_id]["annual_volume"]
                    buyer_revenue[buyer.buyer_id] = buyer_revenue.get(buyer.buyer_id, 0.0) + vol

        total_vol = sum(buyer_revenue.values()) or 1.0
        hhi = sum((v / total_vol) ** 2 for v in buyer_revenue.values())
        hhi_classification = (
            "LOW" if hhi < 0.15 else "MODERATE" if hhi < 0.25 else "HIGH"
        )

        # Overdue analysis
        overdue_by_buyer: dict[str, float] = {}
        for inv in self._world._invoices.values():
            if inv["status"] in ("pending", "overdue"):
                if (day - inv.get("acceptance_day", inv["issue_day"])) > 45:
                    bid = inv["buyer_id"]
                    overdue_by_buyer[bid] = overdue_by_buyer.get(bid, 0.0) + inv["amount"]

        # Top 3 risks
        risks = []
        if hhi > 0.25:
            risks.append({
                "risk": "HIGH_CONCENTRATION",
                "severity": "HIGH",
                "detail": f"HHI={hhi:.3f}: top buyer controls >{int(max(buyer_revenue.values())/total_vol*100)}% of revenue.",
                "recommended_action": "Diversify buyer portfolio or use TReDS to reduce single-buyer dependence.",
            })
        for bid, amount in sorted(overdue_by_buyer.items(), key=lambda x: -x[1])[:2]:
            risks.append({
                "risk": "BUYER_PAYMENT_DELAY",
                "severity": "MEDIUM",
                "buyer_id": bid,
                "overdue_amount": round(amount, 2),
                "recommended_action": "Consider TReDS discount or prepare Samadhaan case (HARD only).",
            })

        # Solvency risk
        for sme in cfg.smes:
            cash = self._world.get_cash_balance(sme.sme_id)
            od_headroom = self._world.get_overdraft_limit(sme.sme_id) - self._world.get_overdraft_drawn(sme.sme_id)
            if cash + od_headroom < 100_000:
                risks.append({
                    "risk": "SOLVENCY_RISK",
                    "severity": "CRITICAL",
                    "sme_id": sme.sme_id,
                    "net_liquid": round(cash + od_headroom, 2),
                    "recommended_action": "Immediately draw overdraft or discount eligible invoices via TReDS.",
                })

        # Belief-state buyer reliability
        buyer_reliability = self._belief.buyer_reliability_summary()

        return {
            "app": "analytics_app",
            "endpoint": "portfolio_risks",
            "day": day,
            "concentration_hhi": round(hhi, 4),
            "hhi_classification": hhi_classification,
            "buyer_revenue_shares": {
                bid: round(v / total_vol, 3) for bid, v in buyer_revenue.items()
            },
            "top_risks": risks[:5],
            "buyer_reliability_beliefs": buyer_reliability,
            "vendor_stress_beliefs": self._belief.vendor_stress_summary(),
        }

    def scenario_analysis(
        self,
        sme_id: Optional[str] = None,
        buyer_id: Optional[str] = None,
    ) -> dict:
        """
        Simulate three scenarios (base, stress, optimistic) for the next 30 days.
        Uses belief-state uncertainty to bound outcomes.
        """
        sme_id = sme_id or self._world._cfg.smes[0].sme_id
        day = self._world._current_day

        base_proj = self._world.get_projected_cashflow(sme_id, 30)
        cash_now = self._world.get_cash_balance(sme_id)

        # Belief uncertainty bounds
        buyer_reliability_avg = self._belief.aggregate_buyer_risk()
        stress_factor = 1.0 - buyer_reliability_avg * 0.3

        scenarios = {
            "base": {
                "net_cashflow_30d": round(base_proj["net_cashflow"], 2),
                "ending_cash": round(cash_now + base_proj["net_cashflow"], 2),
                "assumptions": "Buyers pay on contracted terms; no defaults.",
            },
            "stress": {
                "net_cashflow_30d": round(base_proj["net_cashflow"] * stress_factor, 2),
                "ending_cash": round(cash_now + base_proj["net_cashflow"] * stress_factor, 2),
                "assumptions": (
                    f"Buyer reliability at lower belief bound; "
                    f"20-30% of receivables delayed by additional 15 days."
                ),
            },
            "optimistic": {
                "net_cashflow_30d": round(base_proj["net_cashflow"] * 1.15, 2),
                "ending_cash": round(cash_now + base_proj["net_cashflow"] * 1.15, 2),
                "assumptions": "All buyers pay 10 days early; vendor supply unchanged.",
            },
        }

        solvency_at_risk = scenarios["stress"]["ending_cash"] < 0

        return {
            "app": "analytics_app",
            "endpoint": "scenario_analysis",
            "sme_id": sme_id,
            "buyer_id": buyer_id,
            "current_day": day,
            "cash_now": round(cash_now, 2),
            "scenarios": scenarios,
            "solvency_at_risk_in_stress": solvency_at_risk,
            "recommended_action": (
                "Draw overdraft or discount invoice immediately to prevent stress-case insolvency."
                if solvency_at_risk
                else "Cash position appears resilient in base and stress cases."
            ),
        }

    def kpi_dashboard(self, sme_id: Optional[str] = None) -> dict:
        """Primary KPI aggregator — maps to the agent's observable state."""
        sme_id = sme_id or self._world._cfg.smes[0].sme_id
        kpis = self._world.compute_kpis(sme_id)

        total_financing_cost = self._world.get_total_financing_cost()
        total_revenue = sum(self._world._revenue_collected.values())
        financing_cost_pct = (
            (total_financing_cost / total_revenue * 100) if total_revenue > 0 else 0.0
        )

        return {
            "app": "analytics_app",
            "endpoint": "kpi_dashboard",
            "sme_id": sme_id,
            "day": self._world._current_day,
            **kpis,
            "total_financing_cost": round(total_financing_cost, 2),
            "total_revenue_collected": round(total_revenue, 2),
            "financing_cost_pct_of_revenue": round(financing_cost_pct, 3),
            "current_interest_rate_annual_pct": round(
                self._world._rate_model.current_rate * 100, 3
            ),
        }
