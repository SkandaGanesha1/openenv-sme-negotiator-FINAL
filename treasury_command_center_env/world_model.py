"""
LatentWorldModel — DreamerV3-inspired symbolic RSSM for treasury world modeling.

Architecture (symbolic, no neural network — runs inside the environment):
  Deterministic component h_t:  EMA of observable KPI history
  Stochastic component z_t:     Gaussian noise around h_t (models uncertainty)
  Transition model:             linear extrapolation of KPI trends
  Prediction head:              h_t + z_t → predicted KPIs at t+k

Purpose:
  1. Plan N steps ahead without running the full SimPy simulation
     (agent calls analytics_app.scenario_analysis; this model gives a second,
      independent estimate visible in the observation)
  2. Compute belief entropy — how uncertain is the world model? (exposed in obs)
  3. Reward shaping bonus — when the model predicted well, reward exploration
     of novel strategies; when wrong, signal high uncertainty to guide queries

DreamerV3 analogies:
  h_t  = deterministic recurrent state (GRU hidden in real DreamerV3)
         → here: weighted EMA of [cash_buffer, dso, vendor_stress, hhi]
  z_t  = stochastic discrete latent (categorical in real DreamerV3)
         → here: Gaussian(h_t, σ_t) where σ_t tracks residual variance
  RSSM = Recurrent State-Space Model
         → here: symbolic linear recurrence with learned EMA coefficients
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Tuple


_KPI_KEYS = ["cash_buffer_days", "dso_days", "vendor_stress_score", "concentration_risk_hhi"]
_EMA_ALPHA = 0.3          # smoothing factor for deterministic state h_t
_SIGMA_FLOOR = 0.5        # minimum std for stochastic component z_t
_SIGMA_DECAY = 0.95       # σ shrinks as predictions improve
_SIGMA_INFLATE = 1.10     # σ grows when prediction is badly wrong
_HORIZON = 30             # planning horizon in days


class LatentWorldModel:
    """
    Symbolic RSSM that tracks a deterministic hidden state over treasury KPIs
    and produces planning lookaheads + uncertainty estimates.

    State vectors (all float):
      h: deterministic EMA state, shape [4]  (one per KPI_KEY)
      sigma: per-KPI predictive uncertainty, shape [4]

    Metrics exposed to environment:
      belief_entropy          — Shannon entropy of z_t distribution (bits)
      predicted_cash_buffer   — h_t[0] + drift extrapolation for 30 days
      prediction_error        — MAE between last prediction and observed KPI
    """

    def __init__(self) -> None:
        self._h: Dict[str, float] = {k: 0.0 for k in _KPI_KEYS}
        self._sigma: Dict[str, float] = {k: 10.0 for k in _KPI_KEYS}
        self._last_prediction: Dict[str, float] = {}
        self._history: deque[Dict[str, float]] = deque(maxlen=20)
        self._step: int = 0
        self._cumulative_error: float = 0.0
        self._prediction_errors: List[float] = []

    # ── Update (observe new KPIs) ──────────────────────────────────────────────

    def observe(self, kpis: Dict[str, float]) -> None:
        """
        Update deterministic state h_t with new observation (EMA update).
        Also update σ_t based on how well the last prediction matched.
        """
        # Compute prediction error vs last prediction
        if self._last_prediction and self._step > 0:
            errors = [
                abs(kpis.get(k, 0.0) - self._last_prediction.get(k, 0.0))
                for k in _KPI_KEYS
            ]
            mae = sum(errors) / len(errors)
            self._prediction_errors.append(mae)
            self._cumulative_error += mae

            # Update σ per-KPI: shrink when error is small, inflate when large
            for k in _KPI_KEYS:
                err_k = abs(kpis.get(k, 0.0) - self._last_prediction.get(k, 0.0))
                norm_err = err_k / max(abs(kpis.get(k, 1.0)), 1.0)
                if norm_err < 0.10:
                    self._sigma[k] = max(_SIGMA_FLOOR, self._sigma[k] * _SIGMA_DECAY)
                elif norm_err > 0.30:
                    self._sigma[k] *= _SIGMA_INFLATE

        # EMA update to deterministic state h_t
        for k in _KPI_KEYS:
            obs_val = kpis.get(k, 0.0)
            if self._step == 0:
                self._h[k] = obs_val
            else:
                self._h[k] = (1.0 - _EMA_ALPHA) * self._h[k] + _EMA_ALPHA * obs_val

        self._history.append({k: kpis.get(k, 0.0) for k in _KPI_KEYS})
        self._step += 1

    # ── Predict (generate lookahead) ───────────────────────────────────────────

    def predict(self, horizon_days: int = _HORIZON) -> Dict[str, float]:
        """
        Project h_t forward by `horizon_days` using linear trend extrapolation.
        Prediction is stored for error computation on next observe() call.

        Returns predicted KPI dict with uncertainty bounds.
        """
        trend = self._compute_trend()
        predicted: Dict[str, float] = {}
        for k in _KPI_KEYS:
            extrapolated = self._h[k] + trend[k] * horizon_days
            # Apply domain constraints
            if k == "vendor_stress_score":
                extrapolated = max(0.0, min(1.0, extrapolated))
            elif k == "concentration_risk_hhi":
                extrapolated = max(0.0, min(1.0, extrapolated))
            elif k == "cash_buffer_days":
                extrapolated = max(-999.0, extrapolated)
            elif k == "dso_days":
                extrapolated = max(0.0, extrapolated)
            predicted[k] = round(extrapolated, 3)

        self._last_prediction = dict(predicted)
        return predicted

    def _compute_trend(self) -> Dict[str, float]:
        """Linear trend (slope) over recent history window."""
        if len(self._history) < 2:
            return {k: 0.0 for k in _KPI_KEYS}

        history_list = list(self._history)
        n = len(history_list)
        trend: Dict[str, float] = {}
        for k in _KPI_KEYS:
            vals = [h[k] for h in history_list]
            # Simple linear regression slope
            mean_x = (n - 1) / 2.0
            mean_y = sum(vals) / n
            numer = sum((i - mean_x) * (vals[i] - mean_y) for i in range(n))
            denom = sum((i - mean_x) ** 2 for i in range(n))
            trend[k] = numer / max(denom, 1e-9)
        return trend

    # ── Uncertainty / entropy ──────────────────────────────────────────────────

    def belief_entropy(self) -> float:
        """
        Shannon entropy of the z_t distribution (bits).

        For a Gaussian with std σ: H = 0.5 * ln(2πeσ²) bits.
        We average across KPI dimensions.

        Higher entropy → agent should query more (observe before acting).
        Lower entropy → model is confident; agent can plan more.
        """
        total_entropy = 0.0
        for k in _KPI_KEYS:
            sigma = max(self._sigma[k], _SIGMA_FLOOR)
            # Gaussian entropy in bits
            h_k = 0.5 * math.log2(2 * math.pi * math.e * sigma ** 2)
            total_entropy += h_k
        return round(total_entropy / len(_KPI_KEYS), 4)

    def prediction_error(self) -> float:
        """MAE of last prediction (0.0 if no prediction made yet)."""
        if not self._prediction_errors:
            return 0.0
        # Use exponential window: recent errors matter more
        n = len(self._prediction_errors)
        weights = [_EMA_ALPHA * (1 - _EMA_ALPHA) ** (n - i - 1) for i in range(n)]
        total_w = sum(weights)
        return sum(w * e for w, e in zip(weights, self._prediction_errors)) / max(total_w, 1e-9)

    def avg_prediction_error(self) -> float:
        if not self._prediction_errors:
            return 0.0
        return sum(self._prediction_errors) / len(self._prediction_errors)

    # ── Planning lookahead (for agent observation) ─────────────────────────────

    def cash_buffer_forecast_30d(self) -> float:
        """
        Predicted cash_buffer_days 30 days from now.
        Used in the agent's observation to plan ahead.
        """
        pred = self.predict(horizon_days=30)
        return pred.get("cash_buffer_days", self._h.get("cash_buffer_days", 0.0))

    def uncertainty_bounds(self, key: str, horizon_days: int = 30) -> Tuple[float, float]:
        """
        95% confidence interval on the prediction for `key`.
        Returns (lower_bound, upper_bound).
        """
        pred = self.predict(horizon_days)
        mu = pred.get(key, 0.0)
        sigma = self._sigma.get(key, _SIGMA_FLOOR)
        # 1.96-sigma for 95% CI
        lb = mu - 1.96 * sigma * math.sqrt(horizon_days / 30.0)
        ub = mu + 1.96 * sigma * math.sqrt(horizon_days / 30.0)
        return (round(lb, 2), round(ub, 2))

    # ── Summary ────────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, float]:
        """Compact summary dict for inclusion in the observation aux."""
        pred = self.predict(30)
        return {
            "latent_h_cash_buffer": round(self._h.get("cash_buffer_days", 0.0), 2),
            "latent_h_dso": round(self._h.get("dso_days", 0.0), 2),
            "latent_h_vendor_stress": round(self._h.get("vendor_stress_score", 0.0), 4),
            "latent_h_hhi": round(self._h.get("concentration_risk_hhi", 0.0), 4),
            "predicted_cash_buffer_30d": pred.get("cash_buffer_days", 0.0),
            "belief_entropy_bits": self.belief_entropy(),
            "prediction_error_mae": round(self.prediction_error(), 4),
            "world_model_steps": self._step,
        }

    def reset(self) -> None:
        self.__init__()
