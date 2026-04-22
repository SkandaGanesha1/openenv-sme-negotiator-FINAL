"""Stochastic process generators for invoice arrivals, amounts, and delays."""

from __future__ import annotations

import math
from random import Random


class InvoiceGenerator:
    """
    Generates batches of invoices using realistic stochastic processes:
    - Arrival count: Poisson(λ * days/30)
    - Amount: Lognormal(μ, σ) → right-skewed INR values
    - Delay: Lognormal(μ, σ) → days until buyer pays
    """

    def __init__(
        self,
        rng: Random,
        lambda_per_month: float,
        amount_mu: float,
        amount_sigma: float,
        delay_mu: float,
        delay_sigma: float,
    ) -> None:
        self._rng = rng
        self._lambda = lambda_per_month
        self._amount_mu = amount_mu
        self._amount_sigma = amount_sigma
        self._delay_mu = delay_mu
        self._delay_sigma = delay_sigma

    def _poisson(self, lam: float) -> int:
        """Box-Muller approximation for Poisson sampling via Random (no numpy)."""
        # Knuth algorithm for small λ; normal approx for large λ
        if lam < 30:
            L = math.exp(-lam)
            k = 0
            p = 1.0
            while p > L:
                k += 1
                p *= self._rng.random()
            return k - 1
        # Normal approximation for large λ
        return max(0, round(self._rng.gauss(lam, math.sqrt(lam))))

    def _lognormal(self, mu: float, sigma: float) -> float:
        return math.exp(self._rng.gauss(mu, sigma))

    def sample_invoice_count(self, days: int) -> int:
        lam = self._lambda * (days / 30.0)
        return self._poisson(lam)

    def sample_amount(self) -> float:
        """INR invoice face value."""
        return round(self._lognormal(self._amount_mu, self._amount_sigma) * 1000, 2)

    def sample_delay(self) -> int:
        """Days until payment from invoice acceptance date."""
        return max(1, int(self._lognormal(self._delay_mu, self._delay_sigma)))

    def generate_batch(
        self,
        days: int,
        sme_id: str,
        buyer_id: str,
        start_day: int,
        invoice_id_offset: int = 0,
    ) -> list[dict]:
        count = self.sample_invoice_count(days)
        invoices = []
        for i in range(count):
            issue_day = start_day + int(self._rng.random() * days)
            amount = self.sample_amount()
            delay = self.sample_delay()
            invoices.append(
                {
                    "invoice_id": f"{sme_id}_{buyer_id}_{invoice_id_offset + i:04d}",
                    "sme_id": sme_id,
                    "buyer_id": buyer_id,
                    "amount": amount,
                    "issue_day": issue_day,
                    "due_day": issue_day + delay,
                    "status": "pending",  # pending | paid | discounted | overdue
                    "payment_day": None,
                    "treds_discounted": False,
                    "dd_scheme": False,
                    "overdue_flagged": False,
                }
            )
        return invoices


class InterestRateModel:
    """
    Vasicek-inspired mean-reverting interest rate model.
    Used for overdraft and bridging finance cost computation.
    """

    def __init__(
        self,
        rng: Random,
        base_rate: float = 0.18,
        mean_reversion_speed: float = 0.10,
        long_run_rate: float = 0.18,
        volatility: float = 0.03,
    ) -> None:
        self._rng = rng
        self._rate = base_rate
        self._kappa = mean_reversion_speed
        self._theta = long_run_rate
        self._sigma = volatility

    def advance(self, dt: float = 1 / 365) -> float:
        """One Euler step of Vasicek SDE: dr = κ(θ−r)dt + σ·dW."""
        dW = self._rng.gauss(0, math.sqrt(dt))
        self._rate += self._kappa * (self._theta - self._rate) * dt + self._sigma * dW
        self._rate = max(0.05, min(0.35, self._rate))
        return self._rate

    @property
    def current_rate(self) -> float:
        return self._rate

    def cost_of_bridging(self, amount: float, days: int) -> float:
        """Simple interest: amount × rate × days/365."""
        return round(amount * self._rate * (days / 365.0), 2)
