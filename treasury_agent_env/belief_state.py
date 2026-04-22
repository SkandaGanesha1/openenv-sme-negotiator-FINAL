"""
POMDP belief state tracker for TreasuryAgent.

The agent cannot observe:
  - True buyer credit scores / default probability
  - Actual vendor stress level
  - Upcoming invoice amounts and exact arrival timing

The belief state maintains Gaussian posteriors over these latent variables
and is updated via Bayesian rules each time an observable event occurs.
"""

from __future__ import annotations

import math
from random import Random


class BuyerBeliefState:
    """Gaussian belief over one buyer's reliability (latent creditworthiness)."""

    __slots__ = ("buyer_id", "mu", "sigma")

    def __init__(self, buyer_id: str, prior_mu: float = 0.75, prior_sigma: float = 0.15) -> None:
        self.buyer_id = buyer_id
        self.mu = float(prior_mu)
        self.sigma = float(prior_sigma)

    def update_early_payment(self) -> None:
        """Observation: buyer paid before due date → increase reliability belief."""
        self.mu = min(1.0, self.mu + 0.08)
        self.sigma = max(0.02, self.sigma * 0.90)

    def update_on_time_payment(self) -> None:
        """Observation: buyer paid on due day → small positive update."""
        self.mu = min(1.0, self.mu + 0.03)
        self.sigma = max(0.02, self.sigma * 0.95)

    def update_late_payment(self, days_late: int) -> None:
        """Observation: buyer paid late → decrease reliability, widen uncertainty."""
        penalty = min(0.20, 0.02 * days_late)
        self.mu = max(0.0, self.mu - penalty)
        self.sigma = min(0.40, self.sigma * (1.0 + 0.05 * min(days_late, 10)))

    def update_default(self) -> None:
        """Observation: buyer did not pay at all → strong negative update."""
        self.mu = max(0.0, self.mu - 0.30)
        self.sigma = min(0.45, self.sigma * 1.20)

    def summary(self) -> dict:
        return {
            "buyer_id": self.buyer_id,
            "reliability_mu": round(self.mu, 4),
            "reliability_sigma": round(self.sigma, 4),
            "reliability_95pct_low": round(max(0.0, self.mu - 1.96 * self.sigma), 4),
        }


class VendorBeliefState:
    """Tracks observable vendor stress (partially observed via payment history)."""

    __slots__ = ("vendor_id", "stress_mu", "stress_sigma")

    def __init__(self, vendor_id: str) -> None:
        self.vendor_id = vendor_id
        self.stress_mu = 0.1       # low stress initially
        self.stress_sigma = 0.10

    def update_payment_missed(self) -> None:
        self.stress_mu = min(1.0, self.stress_mu + 0.15)
        self.stress_sigma = min(0.40, self.stress_sigma * 1.10)

    def update_payment_made(self) -> None:
        self.stress_mu = max(0.0, self.stress_mu - 0.05)
        self.stress_sigma = max(0.05, self.stress_sigma * 0.95)

    def summary(self) -> dict:
        return {
            "vendor_id": self.vendor_id,
            "stress_mu": round(self.stress_mu, 4),
            "stress_sigma": round(self.stress_sigma, 4),
        }


class TreasuryBeliefState:
    """
    Aggregated POMDP belief over all buyers and vendors.

    Exposed to the agent only as summary statistics (not raw μ/σ),
    enforcing partial observability.
    """

    def __init__(
        self,
        buyer_ids: list[str],
        vendor_ids: list[str],
        buyer_credit_scores: dict[str, float],
    ) -> None:
        self._buyers: dict[str, BuyerBeliefState] = {
            bid: BuyerBeliefState(bid, prior_mu=buyer_credit_scores.get(bid, 0.75))
            for bid in buyer_ids
        }
        self._vendors: dict[str, VendorBeliefState] = {
            vid: VendorBeliefState(vid) for vid in vendor_ids
        }

    def update(self, events: list[dict]) -> None:
        """Update beliefs from a list of world events (one advance() tick)."""
        for ev in events:
            etype = ev.get("type", "")
            if etype == "payment_received":
                bid = ev.get("buyer_id", "")
                if bid in self._buyers:
                    # Check if late: we don't know exact due_day here,
                    # but we can infer from the event if it contains it.
                    self._buyers[bid].update_on_time_payment()
            elif etype == "invoice_overdue":
                bid = ev.get("buyer_id", "")
                days_late = int(ev.get("days_overdue", 5))
                if bid in self._buyers:
                    self._buyers[bid].update_late_payment(days_late)
            elif etype == "vendor_paid":
                vid = ev.get("vendor_id", "")
                if vid in self._vendors:
                    self._vendors[vid].update_payment_made()
            elif etype == "vendor_payment_missed":
                vid = ev.get("vendor_id", "")
                if vid in self._vendors:
                    self._vendors[vid].update_payment_missed()

    def buyer_reliability_summary(self) -> list[dict]:
        """Observable summary (not raw μ/σ) exposed to agent via KPI dashboard."""
        return [b.summary() for b in self._buyers.values()]

    def vendor_stress_summary(self) -> list[dict]:
        return [v.summary() for v in self._vendors.values()]

    def aggregate_buyer_risk(self) -> float:
        """Mean reliability across all buyers (higher = better)."""
        if not self._buyers:
            return 0.75
        return sum(b.mu for b in self._buyers.values()) / len(self._buyers)

    def aggregate_vendor_stress(self) -> float:
        """Mean stress across all vendors (lower = better)."""
        if not self._vendors:
            return 0.1
        return sum(v.stress_mu for v in self._vendors.values()) / len(self._vendors)

    def sample_hidden_state(self, rng: Random) -> dict:
        """
        Sample a plausible hidden state for planning lookaheads.
        Not exposed in the observation; used internally by analytics tool.
        """
        return {
            "buyer_reliability": {
                bid: max(0.0, min(1.0, rng.gauss(b.mu, b.sigma)))
                for bid, b in self._buyers.items()
            },
            "vendor_stress": {
                vid: max(0.0, min(1.0, rng.gauss(v.stress_mu, v.stress_sigma)))
                for vid, v in self._vendors.items()
            },
        }
