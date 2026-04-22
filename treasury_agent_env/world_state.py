"""
TreasuryWorldState — SimPy discrete-event simulation backed by a NetworkX
supply-chain graph.  This is the hidden ground truth that drives the POMDP;
the agent only observes aggregated KPIs, not this object directly.
"""

from __future__ import annotations

import math
from random import Random
from typing import Optional

import networkx as nx
import simpy

from .stochastic import InvoiceGenerator, InterestRateModel
from .task_config import TreasuryTaskConfig


class TreasuryWorldState:
    """
    Encapsulates the full hidden world state for one episode.

    Timeline: integer days (SimPy time unit = 1 day).
    The environment calls advance(1) each step to tick the clock.
    """

    def __init__(self, config: TreasuryTaskConfig, rng: Random) -> None:
        self._cfg = config
        self._rng = rng

        # ── SimPy environment ──────────────────────────────────────────────
        self._sim = simpy.Environment()
        self._current_day: int = 0

        # ── Supply-chain graph ─────────────────────────────────────────────
        # Directed: vendor → SME → buyer (payment flow direction)
        self._graph = nx.DiGraph()
        self._build_graph()

        # ── Ledgers ───────────────────────────────────────────────────────
        # sme_id → current cash balance
        self._cash: dict[str, float] = {
            s.sme_id: self._rng.uniform(
                s.initial_cash * 0.9, s.initial_cash * 1.1
            )
            for s in config.smes
        }
        # sme_id → overdraft drawn
        self._overdraft_drawn: dict[str, float] = {s.sme_id: 0.0 for s in config.smes}
        # sme_id → accrued overdraft interest
        self._overdraft_interest: dict[str, float] = {s.sme_id: 0.0 for s in config.smes}

        # ── Invoice ledger ─────────────────────────────────────────────────
        self._invoices: dict[str, dict] = {}  # invoice_id → invoice dict
        self._invoice_counter: int = 0

        # ── Interest rate model (shared, one per episode) ──────────────────
        self._rate_model = InterestRateModel(
            rng=rng,
            base_rate=sum(s.overdraft_rate_annual for s in config.smes) / len(config.smes),
        )

        # ── Event log (collected each advance()) ───────────────────────────
        self._event_log: list[dict] = []

        # ── Solvency tracker ───────────────────────────────────────────────
        self._solvency_ok: bool = True
        self._solvency_breach_day: Optional[int] = None

        # ── Buyer DD schemes ───────────────────────────────────────────────
        # buyer_id → dd params if scheme activated
        self._dd_schemes: dict[str, dict] = {}

        # ── Compliance flags ───────────────────────────────────────────────
        # invoice_id → True if Samadhaan case filed
        self._samadhaan_cases: set[str] = set()

        # ── Revenue tracker ───────────────────────────────────────────────
        self._revenue_collected: dict[str, float] = {s.sme_id: 0.0 for s in config.smes}

        # ── Vendor overdue tracking ───────────────────────────────────────
        self._vendor_overdue_days: dict[str, float] = {
            v.vendor_id: 0.0 for v in config.vendors
        }

        # ── Invoice generators ─────────────────────────────────────────────
        self._generators: dict[tuple[str, str], InvoiceGenerator] = {}
        for sme in config.smes:
            for buyer in config.buyers:
                if self._graph.has_edge(sme.sme_id, buyer.buyer_id):
                    lam = self._rng.uniform(
                        config.invoices_per_month_low, config.invoices_per_month_high
                    )
                    self._generators[(sme.sme_id, buyer.buyer_id)] = InvoiceGenerator(
                        rng=rng,
                        lambda_per_month=lam,
                        amount_mu=config.invoice_amount_mu,
                        amount_sigma=config.invoice_amount_sigma,
                        delay_mu=config.payment_delay_mu,
                        delay_sigma=config.payment_delay_sigma,
                    )

    # ── Graph construction ─────────────────────────────────────────────────

    def _build_graph(self) -> None:
        cfg = self._cfg
        for s in cfg.smes:
            self._graph.add_node(
                s.sme_id,
                type="sme",
                overdraft_limit=s.overdraft_limit,
                overdraft_rate=s.overdraft_rate_annual,
                vendor_payment_days=s.vendor_payment_days,
            )
        for b in cfg.buyers:
            self._graph.add_node(
                b.buyer_id,
                type="buyer",
                credit_score=b.credit_score,
                buyer_power=b.buyer_power,
                payment_days=b.payment_days,
                tax_rate=b.tax_rate,
            )
        for v in cfg.vendors:
            self._graph.add_node(
                v.vendor_id,
                type="vendor",
                payment_days_required=v.payment_days_required,
                stress_threshold_days=v.stress_threshold_days,
            )

        # SME → Buyer edges (SME sells to buyer)
        for s in cfg.smes:
            for b in cfg.buyers:
                vol = self._rng.uniform(
                    b.annual_order_volume_low, b.annual_order_volume_high
                )
                self._graph.add_edge(
                    s.sme_id, b.buyer_id,
                    avg_days=b.payment_days,
                    annual_volume=vol,
                    default_risk=round(1.0 - b.credit_score, 3),
                )

        # Vendor → SME edges (vendor supplies SME)
        for v in cfg.vendors:
            for s in cfg.smes:
                supply = self._rng.uniform(
                    v.monthly_supply_low, v.monthly_supply_high
                )
                self._graph.add_edge(
                    v.vendor_id, s.sme_id,
                    payment_days_required=v.payment_days_required,
                    monthly_supply=supply,
                )

    # ── Day advance ─────────────────────────────────────────────────────────

    def advance(self, days: int = 1) -> list[dict]:
        """
        Advance the simulation by `days` steps.
        Returns list of events that occurred (payments received, invoices created, etc.)
        """
        events: list[dict] = []
        for _ in range(days):
            self._current_day += 1
            day = self._current_day

            # 1. Generate new invoices for this day
            events += self._generate_daily_invoices(day)

            # 2. Process due payments from buyers
            events += self._process_buyer_payments(day)

            # 3. Pay vendors
            events += self._process_vendor_payments(day)

            # 4. Accrue overdraft interest
            self._accrue_overdraft_interest()

            # 5. Advance interest rate
            self._rate_model.advance(dt=1 / 365)

            # 6. Check solvency
            events += self._check_solvency(day)

            # 7. Mark overdue invoices
            events += self._mark_overdue(day)

        self._event_log.extend(events)
        return events

    # ── Internal simulation helpers ─────────────────────────────────────────

    def _generate_daily_invoices(self, day: int) -> list[dict]:
        events = []
        for (sme_id, buyer_id), gen in self._generators.items():
            # Use Poisson(λ/30) to determine how many invoices arrive today
            daily_lambda = gen._lambda / 30.0
            count = gen._poisson(daily_lambda)
            for _ in range(count):
                amount = gen.sample_amount()
                delay = gen.sample_delay()
                inv_id = f"{sme_id}_{buyer_id}_{self._invoice_counter:05d}"
                self._invoice_counter += 1
                invoice = {
                    "invoice_id": inv_id,
                    "sme_id": sme_id,
                    "buyer_id": buyer_id,
                    "amount": amount,
                    "issue_day": day,
                    "due_day": day + delay,
                    "status": "pending",
                    "payment_day": None,
                    "treds_discounted": False,
                    "dd_scheme": False,
                    "overdue_flagged": False,
                    "acceptance_day": day,
                }
                self._invoices[inv_id] = invoice
                events.append(
                    {"type": "invoice_created", "day": day, "invoice_id": inv_id,
                     "sme_id": sme_id, "buyer_id": buyer_id, "amount": amount,
                     "due_day": day + delay}
                )
        return events

    def _process_buyer_payments(self, day: int) -> list[dict]:
        events = []
        for inv in list(self._invoices.values()):
            if inv["status"] != "pending":
                continue
            if day < inv["due_day"]:
                continue
            # Buyer pays on due day (with small random late probability)
            buyer_id = inv["buyer_id"]
            buyer_data = self._graph.nodes[buyer_id]
            default_risk = 1.0 - buyer_data["credit_score"]
            # Samadhaan cases improve chance of payment
            samadhaan_bonus = 0.15 if inv["invoice_id"] in self._samadhaan_cases else 0.0
            pay_prob = 1.0 - max(0.0, default_risk - samadhaan_bonus)

            if self._rng.random() < pay_prob:
                sme_id = inv["sme_id"]
                amount = inv["amount"]
                inv["status"] = "paid"
                inv["payment_day"] = day
                self._cash[sme_id] = self._cash.get(sme_id, 0.0) + amount
                self._revenue_collected[sme_id] = (
                    self._revenue_collected.get(sme_id, 0.0) + amount
                )
                # Apply DD scheme discount if active
                if buyer_id in self._dd_schemes and inv.get("dd_scheme"):
                    scheme = self._dd_schemes[buyer_id]
                    discount = amount * scheme.get("discount_rate", 0.0)
                    self._cash[sme_id] -= discount
                events.append(
                    {"type": "payment_received", "day": day, "invoice_id": inv["invoice_id"],
                     "sme_id": sme_id, "buyer_id": buyer_id, "amount": amount}
                )
        return events

    def _process_vendor_payments(self, day: int) -> list[dict]:
        events = []
        for vendor in self._cfg.vendors:
            vid = vendor.vendor_id
            req_days = vendor.payment_days_required
            # SMEs pay vendors on a rolling schedule
            for sme in self._cfg.smes:
                sme_id = sme.sme_id
                if not self._graph.has_edge(vid, sme_id):
                    continue
                edge = self._graph[vid][sme_id]
                monthly_supply = edge["monthly_supply"]
                # Pay approximately 1/30 of monthly supply each day
                daily_payable = monthly_supply / 30.0
                # Due if sme hasn't paid yet (simplified: pay every req_days days)
                if day % req_days == 0:
                    payable = monthly_supply
                    if self._cash.get(sme_id, 0.0) >= payable:
                        self._cash[sme_id] -= payable
                        events.append(
                            {"type": "vendor_paid", "day": day, "vendor_id": vid,
                             "sme_id": sme_id, "amount": payable}
                        )
                    else:
                        # SME can't pay vendor — stress accumulates
                        shortfall = payable - self._cash.get(sme_id, 0.0)
                        self._vendor_overdue_days[vid] = (
                            self._vendor_overdue_days.get(vid, 0.0) + req_days
                        )
                        events.append(
                            {"type": "vendor_payment_missed", "day": day,
                             "vendor_id": vid, "sme_id": sme_id,
                             "shortfall": shortfall}
                        )
        return events

    def _accrue_overdraft_interest(self) -> None:
        for sme in self._cfg.smes:
            drawn = self._overdraft_drawn.get(sme.sme_id, 0.0)
            if drawn > 0:
                daily_rate = sme.overdraft_rate_annual / 365.0
                interest = drawn * daily_rate
                self._overdraft_interest[sme.sme_id] = (
                    self._overdraft_interest.get(sme.sme_id, 0.0) + interest
                )
                # Interest charged to cash daily
                self._cash[sme.sme_id] = self._cash.get(sme.sme_id, 0.0) - interest

    def _check_solvency(self, day: int) -> list[dict]:
        events = []
        for sme_id, cash in self._cash.items():
            od = self._overdraft_drawn.get(sme_id, 0.0)
            od_limit = self._graph.nodes[sme_id]["overdraft_limit"]
            net_position = cash + (od_limit - od)  # available liquidity
            if cash < 0 and self._solvency_ok:
                self._solvency_ok = False
                self._solvency_breach_day = day
                events.append(
                    {"type": "solvency_breach", "day": day,
                     "sme_id": sme_id, "cash": cash}
                )
        return events

    def _mark_overdue(self, day: int) -> list[dict]:
        events = []
        for inv in self._invoices.values():
            if inv["status"] == "pending" and day > inv["due_day"] + 0:
                if not inv["overdue_flagged"]:
                    inv["overdue_flagged"] = True
                    inv["status"] = "overdue"
                    events.append(
                        {"type": "invoice_overdue", "day": day,
                         "invoice_id": inv["invoice_id"],
                         "sme_id": inv["sme_id"], "buyer_id": inv["buyer_id"],
                         "amount": inv["amount"],
                         "days_overdue": day - inv["due_day"]}
                    )
        return events

    # ── Public query methods (used by tool apps) ────────────────────────────

    def get_cash_balance(self, sme_id: str) -> float:
        return self._cash.get(sme_id, 0.0)

    def get_overdraft_drawn(self, sme_id: str) -> float:
        return self._overdraft_drawn.get(sme_id, 0.0)

    def get_overdraft_limit(self, sme_id: str) -> float:
        node = self._graph.nodes.get(sme_id, {})
        return node.get("overdraft_limit", 0.0)

    def get_overdraft_rate(self, sme_id: str) -> float:
        node = self._graph.nodes.get(sme_id, {})
        return node.get("overdraft_rate", 0.18)

    def get_invoices(
        self,
        sme_id: Optional[str] = None,
        buyer_id: Optional[str] = None,
        status: Optional[str] = None,
        window_days: Optional[int] = None,
    ) -> list[dict]:
        result = []
        for inv in self._invoices.values():
            if sme_id and inv["sme_id"] != sme_id:
                continue
            if buyer_id and inv["buyer_id"] != buyer_id:
                continue
            if status and inv["status"] != status:
                continue
            if window_days is not None:
                if inv["issue_day"] < self._current_day - window_days:
                    continue
            result.append(dict(inv))
        return result

    def get_projected_cashflow(self, sme_id: str, window_days: int) -> dict:
        """
        Project cash inflows and outflows over the next `window_days`.
        Inflows = pending invoices due within window.
        Outflows = vendor payables within window (approximation).
        """
        day = self._current_day
        inflows = sum(
            inv["amount"]
            for inv in self._invoices.values()
            if inv["sme_id"] == sme_id
            and inv["status"] in ("pending", "overdue")
            and inv["due_day"] <= day + window_days
        )
        # Approximate vendor outflow
        outflows = 0.0
        for vendor in self._cfg.vendors:
            if self._graph.has_edge(vendor.vendor_id, sme_id):
                edge = self._graph[vendor.vendor_id][sme_id]
                months_in_window = window_days / 30.0
                outflows += edge["monthly_supply"] * months_in_window

        # Subtract overdraft interest
        drawn = self._overdraft_drawn.get(sme_id, 0.0)
        od_rate = self.get_overdraft_rate(sme_id)
        projected_interest = drawn * od_rate * (window_days / 365.0)

        return {
            "sme_id": sme_id,
            "window_days": window_days,
            "projected_inflows": round(inflows, 2),
            "projected_outflows": round(outflows + projected_interest, 2),
            "net_cashflow": round(inflows - outflows - projected_interest, 2),
            "current_cash": round(self._cash.get(sme_id, 0.0), 2),
        }

    def apply_term_update(self, invoice_id: str, new_payment_days: int) -> dict:
        inv = self._invoices.get(invoice_id)
        if inv is None:
            return {"error": f"Invoice {invoice_id} not found"}
        if inv["status"] != "pending":
            return {"error": f"Invoice {invoice_id} status={inv['status']} cannot be updated"}
        old_due = inv["due_day"]
        inv["due_day"] = inv["issue_day"] + new_payment_days
        # Update graph edge avg_days for the buyer
        sme_id, buyer_id = inv["sme_id"], inv["buyer_id"]
        if self._graph.has_edge(sme_id, buyer_id):
            self._graph[sme_id][buyer_id]["avg_days"] = new_payment_days
        return {
            "invoice_id": invoice_id,
            "old_due_day": old_due,
            "new_due_day": inv["due_day"],
            "new_payment_days": new_payment_days,
        }

    def apply_overdraft(self, sme_id: str, amount: float) -> dict:
        od_limit = self.get_overdraft_limit(sme_id)
        od_drawn = self._overdraft_drawn.get(sme_id, 0.0)
        headroom = od_limit - od_drawn
        if amount > headroom:
            return {
                "error": f"Overdraft headroom {headroom:.2f} < requested {amount:.2f}",
                "overdraft_limit": od_limit,
                "already_drawn": od_drawn,
            }
        self._overdraft_drawn[sme_id] = od_drawn + amount
        self._cash[sme_id] = self._cash.get(sme_id, 0.0) + amount
        return {
            "sme_id": sme_id,
            "drawn": amount,
            "total_drawn": self._overdraft_drawn[sme_id],
            "remaining_headroom": headroom - amount,
            "cash_after": self._cash[sme_id],
        }

    def repay_overdraft(self, sme_id: str, amount: float) -> dict:
        od_drawn = self._overdraft_drawn.get(sme_id, 0.0)
        actual = min(amount, od_drawn)
        self._overdraft_drawn[sme_id] = od_drawn - actual
        self._cash[sme_id] = self._cash.get(sme_id, 0.0) - actual
        return {
            "sme_id": sme_id,
            "repaid": actual,
            "remaining_drawn": self._overdraft_drawn[sme_id],
            "cash_after": self._cash[sme_id],
        }

    def apply_treds_discount(self, invoice_id: str, rate: float) -> dict:
        inv = self._invoices.get(invoice_id)
        if inv is None:
            return {"error": f"Invoice {invoice_id} not found"}
        if inv["status"] != "pending":
            return {"error": f"Invoice {invoice_id} not eligible (status={inv['status']})"}
        if inv["treds_discounted"]:
            return {"error": f"Invoice {invoice_id} already discounted"}

        face = inv["amount"]
        days_remaining = max(1, inv["due_day"] - self._current_day)
        discount_amount = face * rate * (days_remaining / 365.0)
        advance = face - discount_amount

        sme_id = inv["sme_id"]
        self._cash[sme_id] = self._cash.get(sme_id, 0.0) + advance
        inv["treds_discounted"] = True
        inv["status"] = "discounted"
        inv["treds_advance"] = advance
        inv["treds_rate"] = rate
        inv["treds_discount_amount"] = discount_amount

        # Financing cost tracking
        self._accrue_financing_cost(sme_id, discount_amount)

        return {
            "invoice_id": invoice_id,
            "face_value": face,
            "discount_rate": rate,
            "days_remaining": days_remaining,
            "discount_amount": round(discount_amount, 2),
            "advance_received": round(advance, 2),
            "cash_after": round(self._cash[sme_id], 2),
        }

    def _accrue_financing_cost(self, sme_id: str, cost: float) -> None:
        """Track total financing cost for grader."""
        key = f"_financing_cost_{sme_id}"
        current = getattr(self, key, 0.0)
        setattr(self, key, current + cost)

    def get_total_financing_cost(self) -> float:
        total = 0.0
        for sme in self._cfg.smes:
            total += getattr(self, f"_financing_cost_{sme.sme_id}", 0.0)
            total += self._overdraft_interest.get(sme.sme_id, 0.0)
        return total

    def activate_dd_scheme(self, buyer_id: str, params: dict) -> dict:
        if buyer_id not in [b.buyer_id for b in self._cfg.buyers]:
            return {"error": f"Buyer {buyer_id} not found"}
        buyer_data = self._graph.nodes[buyer_id]
        buyer_power = buyer_data.get("buyer_power", 0.5)
        target_days = params.get("target_days", buyer_data.get("payment_days", 60))
        max_discount = params.get("max_discount_pct", 0.03)

        # Buyer acceptance probability based on discount offered vs their power
        accept_prob = min(0.9, max_discount / 0.05 * (1.0 - buyer_power * 0.5))
        accepted = self._rng.random() < accept_prob

        if accepted:
            self._dd_schemes[buyer_id] = {
                "target_days": target_days,
                "discount_rate": max_discount,
            }
            # Update graph payment days
            for sme in self._cfg.smes:
                if self._graph.has_edge(sme.sme_id, buyer_id):
                    self._graph[sme.sme_id][buyer_id]["avg_days"] = target_days
        return {
            "buyer_id": buyer_id,
            "accepted": accepted,
            "target_days": target_days,
            "discount_rate": max_discount if accepted else 0.0,
        }

    def flag_samadhaan(self, invoice_id: str) -> dict:
        inv = self._invoices.get(invoice_id)
        if inv is None:
            return {"error": f"Invoice {invoice_id} not found"}
        self._samadhaan_cases.add(invoice_id)
        # Damage relationship: increase buyer_power_score
        buyer_id = inv["buyer_id"]
        if buyer_id in self._graph.nodes:
            bp = self._graph.nodes[buyer_id].get("buyer_power", 0.3)
            self._graph.nodes[buyer_id]["buyer_power"] = min(1.0, bp + 0.10)
        return {
            "invoice_id": invoice_id,
            "samadhaan_filed": True,
            "buyer_id": buyer_id,
            "relationship_impact": "buyer_power_increased_by_0.10",
        }

    # ── KPI aggregation (for observation construction) ─────────────────────

    def compute_kpis(self, primary_sme_id: str) -> dict:
        day = self._current_day
        cfg = self._cfg

        # Cash buffer days
        cash = self._cash.get(primary_sme_id, 0.0)
        daily_burn = 0.0
        for vendor in cfg.vendors:
            if self._graph.has_edge(vendor.vendor_id, primary_sme_id):
                daily_burn += self._graph[vendor.vendor_id][primary_sme_id]["monthly_supply"] / 30.0
        # Add overdraft interest
        od = self._overdraft_drawn.get(primary_sme_id, 0.0)
        od_rate = self.get_overdraft_rate(primary_sme_id)
        daily_burn += od * od_rate / 365.0
        cash_buffer_days = (cash / daily_burn) if daily_burn > 0 else 999.0

        # DSO
        pending_inv = [
            i for i in self._invoices.values()
            if i["sme_id"] == primary_sme_id and i["status"] in ("pending", "overdue")
        ]
        if pending_inv:
            dso = sum(day - i["issue_day"] for i in pending_inv) / len(pending_inv)
        else:
            dso = 0.0

        # Vendor stress (worst vendor)
        max_overdue = max(self._vendor_overdue_days.values()) if self._vendor_overdue_days else 0.0
        vendor_stress = min(1.0, max_overdue / 45.0)

        # HHI concentration risk
        buyer_volumes: dict[str, float] = {}
        for sme in cfg.smes:
            for buyer in cfg.buyers:
                if self._graph.has_edge(sme.sme_id, buyer.buyer_id):
                    vol = self._graph[sme.sme_id][buyer.buyer_id]["annual_volume"]
                    buyer_volumes[buyer.buyer_id] = buyer_volumes.get(buyer.buyer_id, 0.0) + vol
        total_vol = sum(buyer_volumes.values()) or 1.0
        hhi = sum((v / total_vol) ** 2 for v in buyer_volumes.values())

        # Pending invoices
        pending_count = len([i for i in self._invoices.values() if i["status"] == "pending"])
        total_recv = sum(
            i["amount"] for i in self._invoices.values()
            if i["sme_id"] == primary_sme_id and i["status"] in ("pending", "overdue")
        )

        # Upcoming payables (30d)
        upcoming_payables = 0.0
        for vendor in cfg.vendors:
            if self._graph.has_edge(vendor.vendor_id, primary_sme_id):
                upcoming_payables += self._graph[vendor.vendor_id][primary_sme_id]["monthly_supply"]

        # TReDS eligible amount
        treds_eligible = sum(
            i["amount"] for i in self._invoices.values()
            if i["sme_id"] == primary_sme_id
            and i["status"] == "pending"
            and not i["treds_discounted"]
        )

        # Compliance breach count (invoices overdue > 45 days)
        breach_count = sum(
            1 for i in self._invoices.values()
            if i["sme_id"] == primary_sme_id
            and i["status"] in ("pending", "overdue")
            and (day - i.get("acceptance_day", i["issue_day"])) > 45
        )

        return {
            "cash_buffer_days": round(cash_buffer_days, 2),
            "dso_days": round(dso, 2),
            "vendor_stress_score": round(vendor_stress, 4),
            "concentration_risk_hhi": round(hhi, 4),
            "overdraft_used": round(od, 2),
            "overdraft_limit": round(self.get_overdraft_limit(primary_sme_id), 2),
            "pending_invoice_count": pending_count,
            "total_receivables": round(total_recv, 2),
            "upcoming_payables_30d": round(upcoming_payables, 2),
            "treds_eligible_amount": round(treds_eligible, 2),
            "compliance_breach_count": breach_count,
            "solvency_ok": self._solvency_ok,
        }

    def snapshot(self) -> dict:
        """Full hidden state snapshot for grader use."""
        return {
            "day": self._current_day,
            "cash": dict(self._cash),
            "overdraft_drawn": dict(self._overdraft_drawn),
            "overdraft_interest": dict(self._overdraft_interest),
            "revenue_collected": dict(self._revenue_collected),
            "vendor_overdue_days": dict(self._vendor_overdue_days),
            "total_financing_cost": self.get_total_financing_cost(),
            "solvency_ok": self._solvency_ok,
            "solvency_breach_day": self._solvency_breach_day,
            "invoice_count": len(self._invoices),
            "dd_schemes_active": list(self._dd_schemes.keys()),
            "samadhaan_cases": len(self._samadhaan_cases),
        }
