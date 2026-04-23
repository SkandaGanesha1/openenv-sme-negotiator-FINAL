"""
GRPO-compatible multi-objective reward shaper for TreasuryCommandCenter.

RL Techniques integrated:
  RLVR (Reinforcement Learning with Verifiable Rewards):
    - Solvency:     binary verifiable — all SME cash ≥ 0 each day
    - Compliance:   verifiable — zero Section 43B(h) breaches
    - TReDS usage:  verifiable — was TReDS used when it was cheaper than overdraft?

  GRPO (Group Relative Policy Optimization):
    - group_rewards: list of sibling-rollout rewards for the same state
    - normalised_advantage = (r - μ_group) / (σ_group + ε)
    - KL regularisation term exposed for external training loop

  Multi-objective weighted scalarisation (same as base TreasuryAgent):
    1. Solvency       (weight 0.40) — lexicographically first
    2. Financing cost (weight 0.25) — cost ratio vs revenue
    3. Vendor stress  (weight 0.15) — vendor overdue days
    4. Concentration  (weight 0.10) — buyer HHI index
    5. Tool quality   (weight 0.05) — rubric: correct app/endpoint for situation
    6. World-model    (weight 0.05) — prediction accuracy bonus

  Rubrics as Rewards (RaR):
    Per-step rubric evaluates: did the agent query before acting?
    did it use compliance before Samadhaan? did it diversify buyers?

  Constitutional override:
    Any constitutional rule violation → reward capped at 0.05.

Final reward is mapped to strict open interval (0, 1).
"""

from __future__ import annotations

import math
import statistics
from typing import Dict, List, Optional, Tuple

_EPS = 1e-6

_W_SOLVENCY = 0.40
_W_COST = 0.25
_W_VENDOR = 0.15
_W_CONCENTRATION = 0.10
_W_TOOL_QUALITY = 0.05
_W_WORLD_MODEL = 0.05


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


def _strict_unit(x: float) -> float:
    if not math.isfinite(x):
        return _EPS
    return float(min(1.0 - _EPS, max(_EPS, x)))


# ── Verifiable reward components (RLVR) ───────────────────────────────────────

def rlvr_solvency(solvency_ok: bool) -> float:
    """Binary verifiable: all SMEs cash-positive → 1.0, else 0.0."""
    return 1.0 if solvency_ok else 0.0


def rlvr_compliance(compliance_breach_count: int) -> float:
    """Verifiable: zero breaches → 1.0; each breach reduces by 0.2."""
    return _clamp(1.0 - 0.20 * compliance_breach_count)


def rlvr_treds_preference(
    treds_used: bool,
    overdraft_rate: float,
    treds_rate: float,
) -> float:
    """
    Verifiable incentive: when TReDS is cheaper than overdraft, reward using it.
    Binary signal suitable for RLVR.
    """
    if treds_rate <= 0 or overdraft_rate <= 0:
        return 0.5  # neutral when rates unknown
    should_use_treds = treds_rate < overdraft_rate
    if should_use_treds and treds_used:
        return 1.0
    if should_use_treds and not treds_used:
        return 0.2  # penalise missed opportunity
    return 0.8  # used overdraft when it was appropriate


# ── Rubric components (RaR — Rubrics as Rewards) ──────────────────────────────

def rubric_tool_quality(
    app: Optional[str],
    endpoint: Optional[str],
    kpis: Dict[str, float],
    step_in_episode: int,
) -> float:
    """
    Rubric-based reward for tool usage quality.

    Rubric criteria:
    1. First 3 steps of episode: query analytics/ERP before acting (observation phase)
    2. When solvency_ok=False: immediately call bank_app.draw_overdraft or treds_app
    3. When compliance_breach_count > 0: call compliance_app first
    4. Calling analytics_app.kpi_dashboard periodically (every 5 steps)

    Score: fraction of applicable criteria met.
    """
    if not app or not endpoint:
        return 0.3  # no tool call in this step

    score = 0.5  # baseline for any tool call

    solvency_ok = bool(kpis.get("solvency_ok", True))
    breaches = int(kpis.get("compliance_breach_count", 0))
    cash_buffer = float(kpis.get("cash_buffer_days", 999))

    # Criteria 1: observation before action in first 3 steps
    if step_in_episode < 3:
        observational = app in ("analytics_app", "erp_app") and endpoint in (
            "kpi_dashboard", "invoice_summary", "projected_cashflow", "portfolio_risks"
        )
        score += 0.2 if observational else -0.1

    # Criteria 2: solvency crisis → must act on financing
    if not solvency_ok:
        financing_response = (
            (app == "bank_app" and endpoint == "draw_overdraft")
            or (app == "treds_app" and endpoint == "discount_invoice")
        )
        score += 0.3 if financing_response else -0.2

    # Criteria 3: compliance breach → use compliance app
    if breaches > 0:
        compliance_response = app == "compliance_app"
        score += 0.1 if compliance_response else -0.05

    # Criteria 4: periodic KPI check
    if step_in_episode % 5 == 0 and app == "analytics_app":
        score += 0.1

    # Bonus: critical cash buffer — use financing tool
    if cash_buffer < 5.0:
        if app in ("bank_app", "treds_app"):
            score += 0.2

    return _clamp(score)


def rubric_oversight_quality(
    flagged_smes: List[str],
    ground_truth_risky: List[str],
    interventions: Dict[str, str],
) -> float:
    """F1-based rubric for oversight agent precision/recall on risk detection."""
    gt_set = set(ground_truth_risky)
    flagged_set = set(flagged_smes)

    if not gt_set and not flagged_set:
        return 0.75  # correct null detection

    if not gt_set and flagged_set:
        fp_rate = len(flagged_set) / max(len(flagged_set), 1)
        return _clamp(0.5 - 0.3 * fp_rate)

    if gt_set and not flagged_set:
        return _EPS  # missed all risks

    tp = len(gt_set & flagged_set)
    fp = len(flagged_set - gt_set)
    fn = len(gt_set - flagged_set)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, _EPS)

    coverage = sum(1 for sid in flagged_smes if interventions.get(sid, "").strip())
    coverage_rate = coverage / max(len(flagged_smes), 1)

    return _clamp(0.7 * f1 + 0.3 * coverage_rate)


def rubric_manager_quality(
    instructions: Dict[str, str],
    sme_ids_needing_help: List[str],
    query_analytics: Optional[str],
) -> float:
    """Rubric for manager: did it help the right SMEs and query analytics?"""
    score = 0.4  # baseline

    if sme_ids_needing_help:
        helped = sum(1 for sid in sme_ids_needing_help if sid in instructions)
        score += 0.4 * (helped / len(sme_ids_needing_help))

    if query_analytics:
        score += 0.2  # bonus for querying analytics

    return _clamp(score)


# ── Main reward shaper ─────────────────────────────────────────────────────────

class GRPORewardShaper:
    """
    GRPO-compatible multi-objective reward shaper.

    Usage for GRPO training:
      1. Run N rollouts with the same initial observation.
      2. Call compute() on each rollout's final KPIs.
      3. Collect all N raw_rewards into a group_rewards list.
      4. Call normalise_grpo(group_rewards) to get normalised advantages.
      5. Use normalised advantages as the policy gradient signal.

    RLVR path (binary verifiable rewards):
      - Set use_rlvr=True when task has deterministic verifiers.
      - Overrides composite score with hard binary signal when verification fails.
    """

    def __init__(
        self,
        w_solvency: float = _W_SOLVENCY,
        w_cost: float = _W_COST,
        w_vendor: float = _W_VENDOR,
        w_concentration: float = _W_CONCENTRATION,
        w_tool_quality: float = _W_TOOL_QUALITY,
        w_world_model: float = _W_WORLD_MODEL,
    ) -> None:
        assert abs(sum([w_solvency, w_cost, w_vendor, w_concentration, w_tool_quality, w_world_model]) - 1.0) < 1e-5
        self._w = dict(
            solvency=w_solvency,
            cost=w_cost,
            vendor=w_vendor,
            concentration=w_concentration,
            tool_quality=w_tool_quality,
            world_model=w_world_model,
        )

    # ── Component scorers ──────────────────────────────────────────────────────

    def _cost_component(self, total_cost: float, total_revenue: float) -> float:
        if total_revenue <= 0:
            return 0.5
        ratio = total_cost / total_revenue
        return _clamp(1.0 - ratio / 0.10)

    def _vendor_component(self, vendor_stress: float) -> float:
        return _clamp(1.0 - vendor_stress)

    def _concentration_component(self, hhi: float) -> float:
        return _clamp(1.0 - hhi)

    def _world_model_component(self, prediction_error: float) -> float:
        # MAE in cash_buffer_days units: error of 0 → 1.0; error of 30d → 0.0
        return _clamp(1.0 - prediction_error / 30.0)

    # ── Composite reward ───────────────────────────────────────────────────────

    def compute(
        self,
        *,
        solvency_ok: bool,
        total_financing_cost: float,
        total_revenue: float,
        vendor_stress_score: float,
        hhi: float,
        tool_quality_score: float = 0.5,
        world_model_error: float = 0.0,
        compliance_breach_count: int = 0,
        constitution_violated: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Returns (raw_composite, component_dict).

        Constitutional override: any violation caps composite to 0.05.
        Solvency breach: composite capped to 0.09 (catastrophic signal).
        Compliance breaches: deducted from composite.
        """
        s = rlvr_solvency(solvency_ok)
        c = self._cost_component(total_financing_cost, total_revenue)
        v = self._vendor_component(vendor_stress_score)
        k = self._concentration_component(hhi)
        t = _clamp(tool_quality_score)
        m = self._world_model_component(world_model_error)

        composite = (
            self._w["solvency"] * s
            + self._w["cost"] * c
            + self._w["vendor"] * v
            + self._w["concentration"] * k
            + self._w["tool_quality"] * t
            + self._w["world_model"] * m
        )

        # Compliance penalty
        compliance_penalty = 0.05 * compliance_breach_count
        composite = max(0.0, composite - compliance_penalty)

        # Solvency override
        if not solvency_ok:
            composite = min(composite, 0.09)

        # Constitutional override (strongest signal)
        if constitution_violated:
            composite = min(composite, 0.05)

        return composite, {
            "solvency": round(s, 4),
            "financing_cost": round(c, 4),
            "vendor_stress": round(v, 4),
            "concentration": round(k, 4),
            "tool_quality": round(t, 4),
            "world_model": round(m, 4),
            "composite": round(composite, 4),
        }

    def step_reward(
        self,
        *,
        solvency_ok: bool,
        total_financing_cost: float,
        total_revenue: float,
        vendor_stress_score: float,
        hhi: float,
        tool_quality_score: float = 0.5,
        world_model_error: float = 0.0,
        compliance_breach_count: int = 0,
        constitution_violated: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """Per-step partial reward scaled to [0, 0.30]."""
        composite, components = self.compute(
            solvency_ok=solvency_ok,
            total_financing_cost=total_financing_cost,
            total_revenue=total_revenue,
            vendor_stress_score=vendor_stress_score,
            hhi=hhi,
            tool_quality_score=tool_quality_score,
            world_model_error=world_model_error,
            compliance_breach_count=compliance_breach_count,
            constitution_violated=constitution_violated,
        )
        partial = _strict_unit(composite * 0.30)
        return partial, components

    def terminal_reward(
        self,
        *,
        solvency_ok: bool,
        solvency_breach_day: Optional[int],
        max_days: int,
        total_financing_cost: float,
        total_revenue: float,
        vendor_stress_score: float,
        hhi: float,
        tool_quality_score: float = 0.5,
        world_model_error: float = 0.0,
        compliance_breach_count: int = 0,
        constitution_violated: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """Full terminal episode reward in (0, 1)."""
        composite, components = self.compute(
            solvency_ok=solvency_ok,
            total_financing_cost=total_financing_cost,
            total_revenue=total_revenue,
            vendor_stress_score=vendor_stress_score,
            hhi=hhi,
            tool_quality_score=tool_quality_score,
            world_model_error=world_model_error,
            compliance_breach_count=compliance_breach_count,
            constitution_violated=constitution_violated,
        )
        if not solvency_ok and solvency_breach_day is not None:
            fraction = solvency_breach_day / max(1, max_days)
            composite = composite * fraction * 0.5

        terminal = _strict_unit(min(composite, 0.99))
        components["terminal"] = terminal
        return terminal, components

    # ── GRPO normalisation ─────────────────────────────────────────────────────

    @staticmethod
    def normalise_grpo(
        group_rewards: List[float],
        kl_coeff: float = 0.04,
        kl_penalty: float = 0.0,
        eps: float = _EPS,
    ) -> List[float]:
        """
        GRPO group normalisation.

        normalised_advantage_i = (r_i - μ_group) / (σ_group + ε)

        With optional KL penalty (used when training against a reference policy):
          final_i = normalised_advantage_i - kl_coeff * kl_penalty

        Returns list of normalised advantages aligned with input group_rewards.
        """
        if not group_rewards:
            return []
        if len(group_rewards) == 1:
            return [0.0]

        mu = statistics.mean(group_rewards)
        try:
            sigma = statistics.stdev(group_rewards)
        except statistics.StatisticsError:
            sigma = 0.0

        advantages = [(r - mu) / (sigma + eps) for r in group_rewards]
        if kl_penalty > 0:
            advantages = [a - kl_coeff * kl_penalty for a in advantages]
        return advantages

    @staticmethod
    def grpo_success_probability_gain(
        group_rewards: List[float],
        threshold: float = 0.5,
    ) -> float:
        """
        Measures P(success) in the current group — tracks learning progress.

        RLVR theory shows GRPO amplifies P(success) beyond the reference model
        when rewards are binary. This metric tracks that amplification.
        Returns the fraction of group rollouts that exceed `threshold`.
        """
        if not group_rewards:
            return 0.0
        return sum(1 for r in group_rewards if r > threshold) / len(group_rewards)

    # ── Oversight mode ─────────────────────────────────────────────────────────

    def oversight_reward(
        self,
        flagged_smes: List[str],
        ground_truth_risky: List[str],
        interventions: Dict[str, str],
    ) -> Tuple[float, Dict[str, float]]:
        base = rubric_oversight_quality(flagged_smes, ground_truth_risky, interventions)
        gt_set = set(ground_truth_risky)
        flagged_set = set(flagged_smes)
        tp = len(gt_set & flagged_set)
        fp = len(flagged_set - gt_set)
        fn = len(gt_set - flagged_set)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, _EPS)
        return _strict_unit(base), {
            "oversight_f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "composite": round(base, 4),
        }

    # ── Manager mode ───────────────────────────────────────────────────────────

    def manager_reward(
        self,
        *,
        solvent_fraction: float,
        avg_dso_improvement: float,
        gini_days: float,
        total_financing_cost: float,
        total_revenue: float,
        instruction_quality: float = 0.5,
    ) -> Tuple[float, Dict[str, float]]:
        """
        World-level manager reward:
          - solvency fraction of fleet
          - fairness (Gini on payment days, lower = fairer)
          - DSO improvement
          - financing efficiency
        """
        fairness = 1.0 - gini_days
        dso_score = _clamp(avg_dso_improvement / 30.0)
        cost_score = self._cost_component(total_financing_cost, total_revenue)

        composite = (
            0.35 * solvent_fraction
            + 0.25 * fairness
            + 0.20 * dso_score
            + 0.15 * cost_score
            + 0.05 * instruction_quality
        )
        return _strict_unit(composite), {
            "solvent_fraction": round(solvent_fraction, 4),
            "fairness": round(fairness, 4),
            "dso_improvement": round(dso_score, 4),
            "financing_cost": round(cost_score, 4),
            "instruction_quality": round(instruction_quality, 4),
            "composite": round(composite, 4),
        }
