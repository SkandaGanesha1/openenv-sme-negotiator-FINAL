"""Multi-agent world environment for SME payment-term negotiations.

Implements four interaction modes:
  A – Competitive Bidding Market   (SMEs compete for one buyer's contract)
  B – Coalition Bargaining         (SMEs cooperate before negotiating)
  C – Oversight Arena              (OversightAgent supervises fleet of negotiations)
  D – Manager Agent Orchestration  (ManagerAgent instructs a fleet of SME agents)

Each mode is selected via ``task_name`` passed to ``reset()``.
All existing single-agent environments are reused as sub-components.
"""

from __future__ import annotations

import json
import math
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from random import Random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openenv.core import Environment

from sme_negotiator_env.models import NegotiationAction, NegotiationState
from sme_negotiator_env.graders import TASK_GRADERS, grade_task_payment_terms_medium
from server.environment import SMENegotiatorEnvironment

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STRICT_EPS = 1e-6
_DEFAULT_SEED = 42
_MAX_COALITION_MESSAGES = 10
_OVERSIGHT_COMPRESSION_RATIO = 0.3  # fraction of turns shown to oversight agent


# ---------------------------------------------------------------------------
# Enums & Modes
# ---------------------------------------------------------------------------


class WorldMode(str, Enum):
    """Four interaction modes for the multi-agent world."""
    A_COMPETITIVE = "competitive-bidding"
    B_COALITION = "coalition-bargaining"
    C_OVERSIGHT = "oversight-arena"
    D_MANAGER = "manager-orchestration"


TASK_NAME_TO_MODE: Dict[str, WorldMode] = {
    "competitive-bidding": WorldMode.A_COMPETITIVE,
    "coalition-bargaining": WorldMode.B_COALITION,
    "oversight-arena": WorldMode.C_OVERSIGHT,
    "manager-orchestration": WorldMode.D_MANAGER,
}


# ---------------------------------------------------------------------------
# Agent profiles (partial-observability: private fields not exposed to others)
# ---------------------------------------------------------------------------


@dataclass
class SMEProfile:
    """Private + public state for one SME agent."""

    agent_id: str
    # Private (known only to this SME)
    unit_cost: float           # INR/unit manufacturing cost
    monthly_revenue: float     # INR
    liquidity_threshold_days: int  # max receivable days before cash crisis
    interest_rate_annual: float    # cost of short-term borrowing
    cash_balance: float        # current working-capital balance (INR)
    # Public (visible to all)
    industry: str
    reputation_score: float    # 0-1, public proxy for reliability
    # Runtime state
    current_payment_days: int = 90
    current_price: float = 100.0
    is_solvent: bool = True
    treds_enrolled: bool = False
    deal_done: bool = False
    final_deal_days: Optional[int] = None
    final_deal_price: Optional[float] = None
    total_reward: float = 0.0
    # Coalition support
    coalition_agreement_target_days: Optional[int] = None


@dataclass
class BuyerProfile:
    """Private + public state for one buyer agent."""

    agent_id: str
    # Private (SMEs must infer from behavior)
    reservation_price: float       # max willingness-to-pay per unit
    reservation_days: int          # preferred max payment days
    fairness_preference: str       # "exploitative" | "neutral" | "fair"
    # Public
    industry: str
    power_score: float             # 0-1, market power
    contract_volume: int           # units per period
    # Runtime
    accepted_sme_ids: List[str] = field(default_factory=list)
    volume_split: Dict[str, float] = field(default_factory=dict)  # sme_id -> fraction


@dataclass
class CoalitionChannel:
    """Shared chat buffer visible only to SMEs in Mode B."""

    messages: List[Dict[str, str]] = field(default_factory=list)
    max_messages: int = _MAX_COALITION_MESSAGES

    def post(self, sender_id: str, text: str) -> None:
        if len(self.messages) < self.max_messages:
            self.messages.append({"sender": sender_id, "text": text})

    def as_text(self) -> str:
        if not self.messages:
            return "[coalition channel: empty]"
        lines = [f"[{m['sender']}]: {m['text']}" for m in self.messages]
        return "\n".join(lines)


@dataclass
class NegotiationPair:
    """Wraps a single SMENegotiatorEnvironment for one (SME, Buyer) pair."""

    pair_id: str
    sme_id: str
    buyer_id: str
    env: SMENegotiatorEnvironment
    task_name: str = "payment-terms-medium"
    round_count: int = 0
    done: bool = False
    deal_reached: bool = False
    final_days: Optional[int] = None
    final_price: Optional[float] = None
    terminal_reward: float = 0.0
    # Partial observability: store compressed turn history
    turn_history: List[Dict[str, Any]] = field(default_factory=list)

    def compress_history_for_oversight(self) -> List[Dict[str, Any]]:
        """Return a sampled subset of turns (partial observability for OversightAgent)."""
        n = len(self.turn_history)
        sample_size = max(1, int(n * _OVERSIGHT_COMPRESSION_RATIO))
        step = max(1, n // sample_size)
        return self.turn_history[::step][:sample_size]


@dataclass
class WorldState:
    """Aggregated world state across all agents and pairs."""

    episode_id: str
    mode: WorldMode
    world_step: int = 0
    world_horizon: int = 30          # max world steps per episode
    month: int = 0                   # simulated months elapsed
    world_horizon_months: int = 3    # for Mode D
    smes: Dict[str, SMEProfile] = field(default_factory=dict)
    buyers: Dict[str, BuyerProfile] = field(default_factory=dict)
    pairs: Dict[str, NegotiationPair] = field(default_factory=dict)
    coalition: Optional[CoalitionChannel] = None
    # Oversight / Manager specific
    oversight_flags: List[str] = field(default_factory=list)          # pair_ids flagged as unfair
    oversight_interventions: Dict[str, str] = field(default_factory=dict)
    manager_instructions: Dict[str, str] = field(default_factory=dict)  # sme_id -> instruction
    # Metrics accumulated per episode
    metrics: Dict[str, float] = field(default_factory=dict)
    done: bool = False
    termination_reason: str = "ongoing"


# ---------------------------------------------------------------------------
# Scripted buyer policy (used unless replaced by LLM)
# ---------------------------------------------------------------------------


def _scripted_buyer_response(
    buyer: BuyerProfile,
    sme: SMEProfile,
    proposed_price: float,
    proposed_days: int,
    rng: Random,
) -> Tuple[bool, float, int, str]:
    """Return (accept, counter_price, counter_days, rationale).

    Buyer behavior is governed by its fairness_preference:
    - exploitative: maximally extracts concessions
    - neutral: concedes slowly toward reservation values
    - fair: concedes generously if SME is at risk
    """
    fp = buyer.fairness_preference
    jitter = rng.uniform(0.97, 1.03)

    # Check if SME offer is within buyer's reservation
    price_ok = proposed_price <= buyer.reservation_price * jitter
    days_ok = proposed_days <= buyer.reservation_days

    if price_ok and days_ok:
        return True, proposed_price, proposed_days, "Buyer accepts — terms within reservation."

    # Compute counter-offer
    if fp == "exploitative":
        concede_price_frac = rng.uniform(0.005, 0.015)
        concede_days = rng.randint(1, 3)
    elif fp == "fair":
        concede_price_frac = rng.uniform(0.02, 0.05)
        concede_days = rng.randint(5, 10)
        # Fair buyer is more generous when SME cash is low
        if sme.cash_balance < sme.monthly_revenue * 0.5:
            concede_days += rng.randint(3, 7)
    else:  # neutral
        concede_price_frac = rng.uniform(0.01, 0.025)
        concede_days = rng.randint(2, 6)

    counter_price = round(max(sme.unit_cost * 1.02, buyer.reservation_price * (1.0 - concede_price_frac)), 2)
    counter_days = max(20, buyer.reservation_days - concede_days)
    rationale = (
        f"Buyer ({fp}) counters: price={counter_price:.2f}, days={counter_days}."
    )
    return False, counter_price, counter_days, rationale


# ---------------------------------------------------------------------------
# Utility: strict open-interval reward clamping
# ---------------------------------------------------------------------------


def _clamp01(x: float) -> float:
    if not math.isfinite(x):
        return _STRICT_EPS
    return float(min(1.0 - _STRICT_EPS, max(_STRICT_EPS, x)))


def _compute_sme_base_reward(pair: NegotiationPair, sme: SMEProfile) -> float:
    """Reuse the existing single-env terminal grader for this pair."""
    env_state: Optional[NegotiationState] = pair.env.get_state()
    if env_state is None:
        return _STRICT_EPS
    grader = TASK_GRADERS.get(pair.task_name, grade_task_payment_terms_medium)
    return _clamp01(float(grader(env_state)))


def _compute_solvency_penalty(sme: SMEProfile) -> float:
    """Penalty when SME cash balance is below one month of supplier payables."""
    min_safe = sme.monthly_revenue * 0.25
    if sme.cash_balance >= min_safe:
        return 0.0
    shortfall_fraction = (min_safe - sme.cash_balance) / max(min_safe, 1.0)
    return _clamp01(0.1 + 0.3 * shortfall_fraction)


def _update_sme_cash_flow(sme: SMEProfile, agreed_days: int, volume: int, price: float) -> None:
    """Simulate one payment-cycle's cash-flow impact on the SME."""
    receivable = volume * price
    # Working capital cost: opportunity cost of waiting `agreed_days`
    daily_rate = sme.interest_rate_annual / 365.0
    financing_cost = receivable * daily_rate * agreed_days
    # Supplier payment assumed at 30 days
    supplier_payment = sme.monthly_revenue * 0.55  # COGS ~55% of revenue
    net_inflow = receivable - financing_cost - supplier_payment
    sme.cash_balance += net_inflow
    if sme.cash_balance <= 0.0:
        sme.is_solvent = False


# ---------------------------------------------------------------------------
# Observation building helpers (partial observability enforced)
# ---------------------------------------------------------------------------


def _sme_public_view(sme: SMEProfile) -> Dict[str, Any]:
    """Public fields only — no cost/liquidity/cash exposed to competitors."""
    return {
        "agent_id": sme.agent_id,
        "industry": sme.industry,
        "reputation_score": sme.reputation_score,
        "is_solvent": sme.is_solvent,
        "deal_done": sme.deal_done,
    }


def _buyer_public_view(buyer: BuyerProfile) -> Dict[str, Any]:
    """Public fields — reservation price/days and fairness are hidden."""
    return {
        "agent_id": buyer.agent_id,
        "industry": buyer.industry,
        "power_score": buyer.power_score,
        "contract_volume": buyer.contract_volume,
    }


def _build_sme_observation_text(
    sme: SMEProfile,
    buyer: BuyerProfile,
    pair: NegotiationPair,
    world: WorldState,
    *,
    competitor_public_views: Optional[List[Dict[str, Any]]] = None,
    coalition_text: Optional[str] = None,
    manager_instruction: Optional[str] = None,
) -> str:
    """Construct a language-centric observation string for an SME agent."""
    parts: List[str] = []
    parts.append(
        f"[SME {sme.agent_id}] Round {pair.round_count + 1} | Mode: {world.mode.value}"
    )
    parts.append(
        f"Your financials: unit_cost=₹{sme.unit_cost:.2f}, "
        f"monthly_revenue=₹{sme.monthly_revenue:,.0f}, "
        f"liquidity_threshold={sme.liquidity_threshold_days}d, "
        f"cash_balance=₹{sme.cash_balance:,.0f}, "
        f"interest_rate={sme.interest_rate_annual*100:.1f}%/yr"
    )
    parts.append(
        f"Buyer ({buyer.agent_id}): industry={buyer.industry}, "
        f"power_score={buyer.power_score:.2f}, "
        f"volume={buyer.contract_volume} units"
    )
    if pair.turn_history:
        last = pair.turn_history[-1]
        parts.append(
            f"Last buyer counter: price=₹{last.get('buyer_price', '?')}, "
            f"days={last.get('buyer_days', '?')}"
        )
    if competitor_public_views:
        comp_lines = [
            f"  {v['agent_id']}: solvent={v['is_solvent']}, deal_done={v['deal_done']}"
            for v in competitor_public_views
        ]
        parts.append("Competitors (public view only):\n" + "\n".join(comp_lines))
    if coalition_text is not None:
        parts.append("Coalition channel:\n" + coalition_text)
    if manager_instruction:
        parts.append(f"Manager instruction: {manager_instruction}")
    parts.append(
        "Action format (JSON): "
        '{"action_type": "propose"|"accept"|"reject", '
        '"price": <float>, "payment_days": <int>, '
        '"use_treds": <bool>, "reason": "<str>", '
        '"propose_late_payment_penalty_clause": <bool>, '
        '"propose_dynamic_discounting": <bool>, '
        '"dynamic_discount_annual_rate": <float 0-0.95>}'
    )
    if world.mode == WorldMode.B_COALITION:
        parts.append(
            "Coalition action format (optional prefix before negotiation action): "
            '{"coalition_message": "<your message to other SMEs>"}'
        )
    return "\n".join(parts)


def _build_oversight_observation_text(
    world: WorldState,
    compressed_summaries: List[Dict[str, Any]],
) -> str:
    parts: List[str] = [
        f"[OversightAgent] World step {world.world_step} | Active pairs: {len(world.pairs)}",
        "Compressed negotiation summaries (partial view):",
    ]
    for summary in compressed_summaries:
        parts.append(
            f"  Pair {summary['pair_id']}: "
            f"buyer_days={summary.get('buyer_days', '?')}, "
            f"buyer_power={summary.get('buyer_power', '?'):.2f}, "
            f"sme_solvent={summary.get('sme_solvent', True)}, "
            f"liq_gap={summary.get('liquidity_gap', 0):.0f}d, "
            f"round={summary.get('round', 0)}"
        )
    parts.append(
        "Action format (JSON): "
        '{"flag_unfair_cases": ["pair_id_1", ...], '
        '"suggested_interventions": {"pair_id": "<recommendation>"}, '
        '"global_explanation": "<pattern summary>"}'
    )
    return "\n".join(parts)


def _build_manager_observation_text(
    world: WorldState,
    global_metrics: Dict[str, Any],
) -> str:
    solvent = sum(1 for s in world.smes.values() if s.is_solvent)
    total = len(world.smes)
    avg_days = (
        sum(s.current_payment_days for s in world.smes.values()) / max(total, 1)
    )
    parts: List[str] = [
        f"[ManagerAgent] Month {world.month}/{world.world_horizon_months} | "
        f"World step {world.world_step}",
        f"Global metrics: solvent_smes={solvent}/{total}, "
        f"avg_payment_days={avg_days:.1f}, "
        f"total_volume={global_metrics.get('total_volume', 0):.0f}",
        "Per-SME summary (no private costs):",
    ]
    for sme_id, sme in world.smes.items():
        pair = next(
            (p for p in world.pairs.values() if p.sme_id == sme_id), None
        )
        buyer_days_str = str(pair.turn_history[-1].get("buyer_days", "?")) if (pair and pair.turn_history) else "?"
        parts.append(
            f"  {sme_id}: industry={sme.industry}, "
            f"solvent={sme.is_solvent}, "
            f"current_days={sme.current_payment_days}, "
            f"buyer_offer={buyer_days_str}d, "
            f"deal_done={sme.deal_done}"
        )
    parts.append(
        "Action format (JSON): "
        '{"instructions": {"sme_id": "<instruction text>"}, '
        '"query_tool": "<erp|risk_score|treds_rate|null>"}'
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# World-level action and observation Pydantic models
# ---------------------------------------------------------------------------

from pydantic import BaseModel, ConfigDict, Field
from openenv.core import Action, Observation


class WorldAction(Action):
    """Unified action type for the multi-agent world environment.

    The active ``role`` selects which fields are used:
    - sme: uses ``negotiation_action`` (and optionally ``coalition_message``)
    - oversight: uses ``flag_unfair_cases``, ``suggested_interventions``, ``global_explanation``
    - manager: uses ``instructions``, ``query_tool``
    """

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    role: str = Field("sme", description="Role: sme | oversight | manager")
    acting_agent_id: str = Field("", description="ID of the agent taking this action")

    # SME action (Mode A, B, D)
    negotiation_action: Optional[Dict[str, Any]] = Field(
        None,
        description="Serialized NegotiationAction dict for the acting SME",
    )
    coalition_message: Optional[str] = Field(
        None,
        description="Message posted to coalition channel before negotiating (Mode B)",
    )

    # Oversight action (Mode C)
    flag_unfair_cases: List[str] = Field(
        default_factory=list,
        description="pair_ids the oversight agent believes are unfair",
    )
    suggested_interventions: Dict[str, str] = Field(
        default_factory=dict,
        description="Recommended intervention per flagged pair_id",
    )
    global_explanation: str = Field(
        "",
        description="Free-text pattern explanation from oversight agent",
    )

    # Manager action (Mode D)
    instructions: Dict[str, str] = Field(
        default_factory=dict,
        description="sme_id -> instruction text",
    )
    query_tool: Optional[str] = Field(
        None,
        description="Mock ERP/API tool name to query: erp | risk_score | treds_rate",
    )


class WorldObservation(Observation):
    """Observation returned to the principal agent each world step."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    world_step: int
    episode_id: str
    mode: str
    role: str                          # who this obs is for
    acting_agent_id: str
    text: str                          # language-centric description
    reward: float
    done: bool
    aux: Dict[str, Any] = Field(default_factory=dict)  # metrics/metrics for logging
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Mock ERP / tool interface for Mode D
# ---------------------------------------------------------------------------


_ERP_TOOL_RESPONSES: Dict[str, str] = {
    "erp": (
        "ERP snapshot: avg_receivable_days=78, outstanding_invoices=14, "
        "overdraft_utilisation=62%."
    ),
    "risk_score": (
        "Risk module: 2 of {n} SMEs in amber zone (days > threshold), "
        "0 in red."
    ),
    "treds_rate": (
        "TReDS platform: current discounting rate=8.2%/yr, "
        "platform_available=true, onboarding_time=3_days."
    ),
}


def _query_mock_tool(tool_name: str, world: WorldState) -> str:
    if tool_name not in _ERP_TOOL_RESPONSES:
        return f"Unknown tool: {tool_name}"
    template = _ERP_TOOL_RESPONSES[tool_name]
    n_amber = sum(
        1 for s in world.smes.values()
        if s.current_payment_days > s.liquidity_threshold_days
    )
    return template.replace("{n}", str(len(world.smes)))


# ---------------------------------------------------------------------------
# Reward functions per mode
# ---------------------------------------------------------------------------


def _mode_a_sme_reward(
    sme: SMEProfile,
    pair: NegotiationPair,
    all_smes: Dict[str, SMEProfile],
) -> float:
    """Mode A: base reward + competitive bonus for winning better terms than rivals."""
    base = _compute_sme_base_reward(pair, sme)
    solvency_penalty = _compute_solvency_penalty(sme)

    # Competitive bonus: reward SMEs that secure lower payment days than competitors
    competitor_days = [
        s.current_payment_days
        for sid, s in all_smes.items()
        if sid != sme.agent_id
    ]
    if competitor_days and sme.deal_done and sme.final_deal_days is not None:
        avg_comp = sum(competitor_days) / len(competitor_days)
        relative_gain = (avg_comp - sme.final_deal_days) / max(avg_comp, 1.0)
        comp_bonus = _clamp01(0.1 * max(0.0, relative_gain))
    else:
        comp_bonus = 0.0

    raw = base - solvency_penalty + comp_bonus
    return _clamp01(raw)


def _mode_a_buyer_reward(
    buyer: BuyerProfile,
    world: WorldState,
) -> float:
    """Mode A: buyer rewards cost savings and avoids choosing insolvent SMEs."""
    accepted_pairs = [
        p for p in world.pairs.values()
        if p.buyer_id == buyer.agent_id and p.deal_reached
    ]
    if not accepted_pairs:
        return _STRICT_EPS

    # Cost savings vs reservation price
    savings_scores = []
    for pair in accepted_pairs:
        sme = world.smes.get(pair.sme_id)
        if sme is None or sme.final_deal_price is None:
            continue
        saving_frac = (buyer.reservation_price - sme.final_deal_price) / max(buyer.reservation_price, 1.0)
        savings_scores.append(max(0.0, saving_frac))

    avg_savings = sum(savings_scores) / max(len(savings_scores), 1)

    # Penalty if chosen SME is insolvent
    insolvency_penalty = sum(
        0.15 for p in accepted_pairs
        if not world.smes.get(p.sme_id, SMEProfile("", 0, 0, 0, 0, 0, "", 0)).is_solvent
    )

    return _clamp01(avg_savings - insolvency_penalty)


def _mode_b_sme_reward(
    sme: SMEProfile,
    pair: NegotiationPair,
    all_smes: Dict[str, SMEProfile],
    coalition_held: bool,
) -> float:
    """Mode B: base reward + coalition bonus or defection penalty."""
    base = _compute_sme_base_reward(pair, sme)
    solvency_penalty = _compute_solvency_penalty(sme)

    target = sme.coalition_agreement_target_days
    coalition_bonus = 0.0
    defection_penalty = 0.0

    if target is not None and sme.deal_done and sme.final_deal_days is not None:
        improvement = (sme.current_payment_days - sme.final_deal_days) / max(sme.current_payment_days, 1.0)
        if improvement >= 0.3:  # moved ≥30% toward target
            coalition_bonus = _clamp01(0.15 if coalition_held else 0.05)
        elif sme.final_deal_days > target + 20:
            # SME accepted very bad terms while others held out
            defection_penalty = 0.1

    raw = base + coalition_bonus - defection_penalty - solvency_penalty
    return _clamp01(raw)


def _mode_c_oversight_reward(
    flags: List[str],
    interventions: Dict[str, str],
    world: WorldState,
) -> float:
    """Mode C: precision-recall on unfair-case detection."""
    ground_truth_unfair = _detect_ground_truth_unfair(world)

    if not ground_truth_unfair and not flags:
        return _clamp01(0.7)  # correct null detection

    if not ground_truth_unfair and flags:
        false_positive_rate = len(flags) / max(len(world.pairs), 1)
        return _clamp01(0.5 - 0.3 * false_positive_rate)

    if ground_truth_unfair and not flags:
        return _STRICT_EPS  # missed all unfair cases

    tp = len(set(flags) & ground_truth_unfair)
    fp = len(set(flags) - ground_truth_unfair)
    fn = len(ground_truth_unfair - set(flags))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    # Bonus for providing non-empty interventions on flagged cases
    intervention_coverage = sum(
        1 for pid in flags if interventions.get(pid, "").strip()
    ) / max(len(flags), 1)

    return _clamp01(0.7 * f1 + 0.3 * intervention_coverage)


def _detect_ground_truth_unfair(world: WorldState) -> set:
    """Heuristic ground-truth: mark pair as unfair if final days are harmful."""
    unfair: set = set()
    for pair_id, pair in world.pairs.items():
        sme = world.smes.get(pair.sme_id)
        if sme is None:
            continue
        # Unfair if agreed days > SME liquidity threshold, or SME became insolvent
        if pair.deal_reached and pair.final_days is not None:
            if pair.final_days > sme.liquidity_threshold_days:
                unfair.add(pair_id)
        if not sme.is_solvent:
            unfair.add(pair_id)
        # Also unfair if buyer power is very high and SME is distressed
        buyer = world.buyers.get(pair.buyer_id)
        if buyer and buyer.power_score > 0.8 and sme.cash_balance < sme.monthly_revenue * 0.3:
            unfair.add(pair_id)
    return unfair


def _mode_d_manager_reward(world: WorldState) -> float:
    """Mode D: world-level reward for solvency, fairness, and volume growth."""
    total = len(world.smes)
    if total == 0:
        return _STRICT_EPS

    solvent_fraction = sum(1 for s in world.smes.values() if s.is_solvent) / total

    # Fairness: Gini-like measure on payment days (lower = more equal = fairer)
    all_days = [s.current_payment_days for s in world.smes.values()]
    gini = _compute_gini(all_days)
    fairness_score = 1.0 - gini

    # Volume: fraction of deals closed
    deals_done = sum(1 for s in world.smes.values() if s.deal_done)
    volume_fraction = deals_done / total

    # Borrowing cost proxy: avg interest burden (lower is better)
    avg_interest = sum(
        s.interest_rate_annual * max(0, s.current_payment_days - 30) / 365.0
        for s in world.smes.values()
    ) / total
    borrowing_score = max(0.0, 1.0 - avg_interest * 20.0)

    raw = (
        0.35 * solvent_fraction
        + 0.25 * fairness_score
        + 0.20 * volume_fraction
        + 0.20 * borrowing_score
    )
    return _clamp01(raw)


def _compute_gini(values: List[float]) -> float:
    """Compute Gini coefficient for a list of non-negative values."""
    n = len(values)
    if n <= 1:
        return 0.0
    sorted_vals = sorted(max(v, 0.0) for v in values)
    total = sum(sorted_vals)
    if total < 1e-9:
        return 0.0
    cumsum = 0.0
    gini_num = 0.0
    for i, v in enumerate(sorted_vals):
        cumsum += v
        gini_num += (2 * (i + 1) - n - 1) * v
    return gini_num / (n * total)


# ---------------------------------------------------------------------------
# Main environment class
# ---------------------------------------------------------------------------


class SMEMultiAgentWorldEnvironment(Environment):
    """OpenEnv multi-agent world environment for SME payment-term negotiations.

    Modes (selected via ``task_name`` in reset kwargs):
      A – competitive-bidding
      B – coalition-bargaining
      C – oversight-arena
      D – manager-orchestration

    The ``step()`` method accepts a ``WorldAction`` and returns a
    ``WorldObservation`` for the current principal agent.  Scheduling
    is round-robin across active agents within the current mode.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    # ------------------------------------------------------------------ init

    def __init__(self) -> None:
        self._rng = Random(_DEFAULT_SEED)
        self._world: Optional[WorldState] = None
        # Round-robin scheduling: ordered list of (role, agent_id) tuples
        self._schedule: List[Tuple[str, str]] = []
        self._schedule_index: int = 0
        self._cumulative_reward: float = 0.0

    # ---------------------------------------------------------------- state

    @property
    def state(self) -> Optional[Dict[str, Any]]:
        if self._world is None:
            return None
        return {
            "episode_id": self._world.episode_id,
            "mode": self._world.mode.value,
            "world_step": self._world.world_step,
            "month": self._world.month,
            "done": self._world.done,
            "metrics": self._world.metrics,
            "num_smes": len(self._world.smes),
            "num_buyers": len(self._world.buyers),
        }

    # --------------------------------------------------------------- helpers

    def _now_id(self) -> str:
        return uuid.uuid4().hex[:8]

    def _sample_sme(self, idx: int) -> SMEProfile:
        """Sample a realistic SME profile with variability."""
        industries = ["textiles", "auto_parts", "pharma", "electronics", "agri_processing"]
        unit_cost = self._rng.uniform(65.0, 88.0)
        return SMEProfile(
            agent_id=f"sme_{idx}",
            unit_cost=round(unit_cost, 2),
            monthly_revenue=self._rng.uniform(300_000, 900_000),
            liquidity_threshold_days=self._rng.randint(30, 60),
            interest_rate_annual=self._rng.uniform(0.18, 0.26),
            cash_balance=self._rng.uniform(100_000, 600_000),
            industry=self._rng.choice(industries),
            reputation_score=round(self._rng.uniform(0.5, 0.95), 2),
            current_payment_days=90,
            current_price=100.0,
            treds_enrolled=self._rng.random() < 0.3,
        )

    def _sample_buyer(self, idx: int, mode: WorldMode) -> BuyerProfile:
        """Sample a realistic buyer profile."""
        fairness_dist = {
            WorldMode.A_COMPETITIVE: ["exploitative", "exploitative", "neutral"],
            WorldMode.B_COALITION: ["exploitative", "neutral", "fair"],
            WorldMode.C_OVERSIGHT: ["exploitative", "neutral", "fair"],
            WorldMode.D_MANAGER: ["neutral", "neutral", "fair"],
        }
        fairness_options = fairness_dist.get(mode, ["neutral"])
        return BuyerProfile(
            agent_id=f"buyer_{idx}",
            reservation_price=self._rng.uniform(92.0, 106.0),
            reservation_days=self._rng.randint(45, 75),
            fairness_preference=self._rng.choice(fairness_options),
            industry=self._rng.choice(["retail", "fmcg", "automotive", "healthcare"]),
            power_score=round(self._rng.uniform(0.4, 0.92), 2),
            contract_volume=self._rng.randint(500, 8000),
        )

    def _create_pair(
        self,
        sme: SMEProfile,
        buyer: BuyerProfile,
        task_name: str = "payment-terms-medium",
    ) -> NegotiationPair:
        """Create a sub-environment for an (SME, Buyer) negotiation pair."""
        sub_env = SMENegotiatorEnvironment()
        pair_id = f"{sme.agent_id}_x_{buyer.agent_id}"
        pair = NegotiationPair(
            pair_id=pair_id,
            sme_id=sme.agent_id,
            buyer_id=buyer.agent_id,
            env=sub_env,
            task_name=task_name,
        )
        sub_env.reset(
            seed=self._rng.randint(0, 99999),
            task_name=task_name,
            episode_id=pair_id,
        )
        return pair

    def _build_schedule_a(self, world: WorldState) -> List[Tuple[str, str]]:
        """Round-robin schedule: each SME acts in turn, buyer is scripted."""
        return [("sme", sme_id) for sme_id in world.smes]

    def _build_schedule_b(self, world: WorldState) -> List[Tuple[str, str]]:
        """Mode B: coalition message phase then negotiation phase per SME."""
        return [("sme", sme_id) for sme_id in world.smes]

    def _build_schedule_c(self, world: WorldState) -> List[Tuple[str, str]]:
        """Mode C: a single oversight agent acts at each world step."""
        return [("oversight", "oversight_agent")]

    def _build_schedule_d(self, world: WorldState) -> List[Tuple[str, str]]:
        """Mode D: manager agent acts at each world step."""
        return [("manager", "manager_agent")]

    def _advance_pair_with_action(
        self,
        pair: NegotiationPair,
        neg_action_dict: Dict[str, Any],
        sme: SMEProfile,
        buyer: BuyerProfile,
    ) -> float:
        """Apply a negotiation action to a pair's sub-env and return step reward."""
        if pair.done:
            return 0.0

        # Build NegotiationAction from dict, with safe defaults
        try:
            neg_action = NegotiationAction(**neg_action_dict)
        except Exception:
            # Fallback to a neutral propose action
            neg_action = NegotiationAction(
                action_type="propose",
                price=round(sme.unit_cost * 1.15, 2),
                payment_days=sme.liquidity_threshold_days,
            )

        obs = pair.env.step(neg_action)
        pair.round_count += 1

        # Record turn in history
        pair.turn_history.append({
            "round": pair.round_count,
            "sme_price": neg_action.price,
            "sme_days": neg_action.payment_days,
            "buyer_price": float(obs.buyer_price),
            "buyer_days": int(obs.buyer_days),
            "buyer_accepted": bool(obs.buyer_accepted),
            "step_reward": float(obs.step_reward),
        })

        # If buyer accepted or scripted buyer decides to accept/counter
        if obs.buyer_accepted or obs.negotiation_done:
            pair.done = True
            pair.deal_reached = bool(obs.buyer_accepted)
            pair.final_days = int(neg_action.payment_days) if obs.buyer_accepted else None
            pair.final_price = float(neg_action.price) if obs.buyer_accepted else None
            pair.terminal_reward = float(obs.reward)

            if pair.deal_reached and pair.final_days is not None:
                sme.deal_done = True
                sme.final_deal_days = pair.final_days
                sme.final_deal_price = pair.final_price
                sme.current_payment_days = pair.final_days
                buyer.accepted_sme_ids.append(sme.agent_id)
                _update_sme_cash_flow(sme, pair.final_days, buyer.contract_volume, pair.final_price or sme.unit_cost * 1.2)
        else:
            # Scripted buyer counters — update SME's view of current buyer offer
            sme.current_payment_days = int(obs.buyer_days)
            sme.current_price = float(obs.buyer_price)

        return float(obs.step_reward)

    def _scripted_advance_pair(
        self,
        pair: NegotiationPair,
        sme: SMEProfile,
        buyer: BuyerProfile,
    ) -> float:
        """Advance a pair using the scripted heuristic SME policy (for Modes C/D sub-envs)."""
        from sme_negotiator_env.client import choose_action

        env_obs = pair.env._obs_from_state(
            buyer_accepted=False,
            negotiation_done=pair.done,
            step_reward=0.0,
            message="",
            reward=0.0,
            done=pair.done,
            metadata={},
        ) if pair.env._state else None

        if env_obs is None or pair.done:
            return 0.0

        heuristic_action = choose_action(env_obs, pair.round_count)
        return self._advance_pair_with_action(
            pair,
            heuristic_action.model_dump(),
            sme,
            buyer,
        )

    # ------------------------------------------------------ reset & step API

    def reset(
        self,
        seed: Optional[int] = None,
        difficulty: str = "MEDIUM",
        **kwargs,
    ) -> WorldObservation:
        """Reset world episode; mode determined by ``task_name`` kwarg."""

        raw_task = kwargs.get("task_name") or kwargs.get("task") or "competitive-bidding"
        mode = TASK_NAME_TO_MODE.get(str(raw_task).strip(), WorldMode.A_COMPETITIVE)

        # Parse world config from kwargs with sensible defaults
        num_smes: int = int(kwargs.get("num_smes", 3 if mode == WorldMode.A_COMPETITIVE else 3))
        num_buyers: int = int(kwargs.get("num_buyers", 1 if mode in (WorldMode.A_COMPETITIVE, WorldMode.B_COALITION) else 2))
        world_horizon_months: int = int(kwargs.get("world_horizon_months", 3))
        coalition_chat_length: int = int(kwargs.get("coalition_chat_length", _MAX_COALITION_MESSAGES))
        num_parallel_envs: int = int(kwargs.get("num_parallel_envs", 4 if mode == WorldMode.C_OVERSIGHT else num_smes))

        if seed is not None:
            self._rng = Random(int(seed))

        episode_id = str(kwargs.get("episode_id") or f"world_{mode.value}_{self._now_id()}")

        # Choose sub-env task name aligned with difficulty
        sub_task_map = {"easy": "payment-terms-easy", "medium": "payment-terms-medium", "hard": "payment-terms-hard"}
        sub_task = sub_task_map.get(difficulty.lower(), "payment-terms-medium")

        # Sample agents
        smes: Dict[str, SMEProfile] = {}
        for i in range(num_smes):
            s = self._sample_sme(i)
            smes[s.agent_id] = s

        buyers: Dict[str, BuyerProfile] = {}
        for j in range(num_buyers):
            b = self._sample_buyer(j, mode)
            buyers[b.agent_id] = b

        # Set coalition target days (Mode B)
        if mode == WorldMode.B_COALITION:
            coalition_target = self._rng.randint(45, 60)
            for s in smes.values():
                s.coalition_agreement_target_days = coalition_target

        # Create negotiation pairs
        pairs: Dict[str, NegotiationPair] = {}
        if mode in (WorldMode.A_COMPETITIVE, WorldMode.B_COALITION):
            # Each SME negotiates with the first (primary) buyer
            primary_buyer = list(buyers.values())[0]
            for sme in smes.values():
                p = self._create_pair(sme, primary_buyer, sub_task)
                pairs[p.pair_id] = p
        elif mode == WorldMode.C_OVERSIGHT:
            # Parallel fleet of negotiations (scripted both sides)
            oversight_smes: Dict[str, SMEProfile] = {}
            oversight_buyers: Dict[str, BuyerProfile] = {}
            for i in range(num_parallel_envs):
                s = self._sample_sme(i + 100)
                b = self._sample_buyer(i + 100, mode)
                oversight_smes[s.agent_id] = s
                oversight_buyers[b.agent_id] = b
                p = self._create_pair(s, b, sub_task)
                pairs[p.pair_id] = p
            smes = oversight_smes
            buyers = oversight_buyers
        elif mode == WorldMode.D_MANAGER:
            # Each SME gets paired with a buyer (round-robin over buyers)
            buyer_list = list(buyers.values())
            for i, sme in enumerate(smes.values()):
                b = buyer_list[i % len(buyer_list)]
                p = self._create_pair(sme, b, sub_task)
                pairs[p.pair_id] = p

        world = WorldState(
            episode_id=episode_id,
            mode=mode,
            world_step=0,
            world_horizon=max(20, num_smes * 8),
            world_horizon_months=world_horizon_months,
            smes=smes,
            buyers=buyers,
            pairs=pairs,
            coalition=CoalitionChannel(max_messages=coalition_chat_length) if mode == WorldMode.B_COALITION else None,
        )
        self._world = world
        self._cumulative_reward = 0.0

        # Build schedule
        if mode == WorldMode.A_COMPETITIVE:
            self._schedule = self._build_schedule_a(world)
        elif mode == WorldMode.B_COALITION:
            self._schedule = self._build_schedule_b(world)
        elif mode == WorldMode.C_OVERSIGHT:
            self._schedule = self._build_schedule_c(world)
        elif mode == WorldMode.D_MANAGER:
            self._schedule = self._build_schedule_d(world)
        else:
            self._schedule = self._build_schedule_a(world)
        self._schedule_index = 0

        return self._build_observation(reward=0.0, done=False)

    def step(self, action: WorldAction, **kwargs) -> WorldObservation:
        """Process one world action and advance the environment."""
        if self._world is None:
            return self._build_observation(reward=_STRICT_EPS, done=True)

        world = self._world
        if world.done:
            return self._build_observation(reward=0.0, done=True)

        step_reward = 0.0

        # Identify current role and acting agent from schedule
        if self._schedule:
            current_role, current_agent = self._schedule[self._schedule_index % len(self._schedule)]
        else:
            current_role, current_agent = action.role, action.acting_agent_id

        # ---- Mode A: Competitive Bidding ----
        if world.mode == WorldMode.A_COMPETITIVE:
            step_reward = self._step_mode_a(action, current_agent, world)

        # ---- Mode B: Coalition Bargaining ----
        elif world.mode == WorldMode.B_COALITION:
            step_reward = self._step_mode_b(action, current_agent, world)

        # ---- Mode C: Oversight Arena ----
        elif world.mode == WorldMode.C_OVERSIGHT:
            step_reward = self._step_mode_c(action, world)

        # ---- Mode D: Manager Orchestration ----
        elif world.mode == WorldMode.D_MANAGER:
            step_reward = self._step_mode_d(action, world)

        # Advance schedule
        self._schedule_index += 1
        world.world_step += 1

        # Update month (Mode D)
        if world.mode == WorldMode.D_MANAGER:
            steps_per_month = max(1, len(world.smes))
            world.month = world.world_step // steps_per_month

        # Check episode termination
        done = self._check_done(world)
        if done:
            world.done = True
            world.termination_reason = "max_steps_or_all_done"
            step_reward = self._terminal_world_reward(world, step_reward)

        self._cumulative_reward += step_reward
        world.metrics["cumulative_reward"] = self._cumulative_reward
        world.metrics["world_step"] = world.world_step
        world.metrics["gini_days"] = _compute_gini(
            [s.current_payment_days for s in world.smes.values()]
        )
        world.metrics["solvent_fraction"] = (
            sum(1 for s in world.smes.values() if s.is_solvent) / max(len(world.smes), 1)
        )

        return self._build_observation(reward=step_reward, done=done)

    # -------------------------------------------------- mode step handlers

    def _step_mode_a(self, action: WorldAction, sme_id: str, world: WorldState) -> float:
        """Mode A: advance the acting SME's negotiation with buyer."""
        sme = world.smes.get(sme_id)
        if sme is None:
            return _STRICT_EPS

        pair = next((p for p in world.pairs.values() if p.sme_id == sme_id), None)
        if pair is None or pair.done:
            # Skip to next active SME
            return _STRICT_EPS

        buyer = world.buyers.get(pair.buyer_id)
        if buyer is None:
            return _STRICT_EPS

        neg_action_dict = action.negotiation_action or {
            "action_type": "propose",
            "price": round(sme.unit_cost * 1.15, 2),
            "payment_days": sme.liquidity_threshold_days,
        }

        step_rew = self._advance_pair_with_action(pair, neg_action_dict, sme, buyer)
        all_smes = world.smes
        final_reward = _mode_a_sme_reward(sme, pair, all_smes)
        return _clamp01(0.4 * step_rew + 0.6 * final_reward) if pair.done else _clamp01(step_rew)

    def _step_mode_b(self, action: WorldAction, sme_id: str, world: WorldState) -> float:
        """Mode B: optional coalition post then negotiation advance."""
        sme = world.smes.get(sme_id)
        if sme is None:
            return _STRICT_EPS

        # Post coalition message if provided
        if action.coalition_message and world.coalition is not None:
            world.coalition.post(sme_id, action.coalition_message)

        pair = next((p for p in world.pairs.values() if p.sme_id == sme_id), None)
        if pair is None or pair.done:
            return _STRICT_EPS

        buyer = world.buyers.get(pair.buyer_id)
        if buyer is None:
            return _STRICT_EPS

        neg_action_dict = action.negotiation_action or {
            "action_type": "propose",
            "price": round(sme.unit_cost * 1.15, 2),
            "payment_days": sme.coalition_agreement_target_days or sme.liquidity_threshold_days,
        }

        step_rew = self._advance_pair_with_action(pair, neg_action_dict, sme, buyer)

        # Determine if coalition held (majority of SMEs hit target)
        targets_met = sum(
            1 for s in world.smes.values()
            if s.deal_done
            and s.coalition_agreement_target_days is not None
            and s.final_deal_days is not None
            and s.final_deal_days <= s.coalition_agreement_target_days
        )
        coalition_held = targets_met > len(world.smes) // 2

        final_reward = _mode_b_sme_reward(sme, pair, world.smes, coalition_held)
        return _clamp01(0.4 * step_rew + 0.6 * final_reward) if pair.done else _clamp01(step_rew)

    def _step_mode_c(self, action: WorldAction, world: WorldState) -> float:
        """Mode C: advance all pairs with scripted policies; score oversight action."""
        # Advance all pairs one step using scripted policies
        for pair in world.pairs.values():
            if not pair.done:
                sme = world.smes.get(pair.sme_id)
                buyer = world.buyers.get(pair.buyer_id)
                if sme and buyer:
                    self._scripted_advance_pair(pair, sme, buyer)

        # Score oversight action
        return _mode_c_oversight_reward(
            action.flag_unfair_cases,
            action.suggested_interventions,
            world,
        )

    def _step_mode_d(self, action: WorldAction, world: WorldState) -> float:
        """Mode D: apply manager instructions then advance pairs."""
        # Handle tool query (mock ERP/API)
        if action.query_tool:
            tool_result = _query_mock_tool(action.query_tool, world)
            world.metrics["last_tool_result"] = tool_result  # type: ignore[assignment]

        # Distribute manager instructions to SMEs
        for sme_id, instruction in action.instructions.items():
            world.manager_instructions[sme_id] = instruction

        # Advance each pair: instruction is parsed if possible, else heuristic
        for pair in world.pairs.values():
            if pair.done:
                continue
            sme = world.smes.get(pair.sme_id)
            buyer = world.buyers.get(pair.buyer_id)
            if sme is None or buyer is None:
                continue

            instruction = world.manager_instructions.get(sme.agent_id, "")
            neg_action_dict = _parse_manager_instruction_to_action(instruction, sme, pair)
            self._advance_pair_with_action(pair, neg_action_dict, sme, buyer)

        return _mode_d_manager_reward(world)

    # ------------------------------------------- termination & rewards

    def _check_done(self, world: WorldState) -> bool:
        if world.world_step >= world.world_horizon:
            return True
        if world.mode == WorldMode.D_MANAGER and world.month >= world.world_horizon_months:
            return True
        # All pairs done
        if world.pairs and all(p.done for p in world.pairs.values()):
            return True
        return False

    def _terminal_world_reward(self, world: WorldState, last_step_reward: float) -> float:
        """Compute final terminal reward blend."""
        if world.mode == WorldMode.A_COMPETITIVE:
            # Average SME reward across all pairs
            rewards = [
                _mode_a_sme_reward(world.smes[p.sme_id], p, world.smes)
                for p in world.pairs.values()
                if p.sme_id in world.smes
            ]
            return _clamp01(sum(rewards) / max(len(rewards), 1)) if rewards else last_step_reward

        elif world.mode == WorldMode.B_COALITION:
            targets_met = sum(
                1 for s in world.smes.values()
                if s.deal_done and s.final_deal_days is not None
                and s.coalition_agreement_target_days is not None
                and s.final_deal_days <= s.coalition_agreement_target_days
            )
            coalition_held = targets_met > len(world.smes) // 2
            rewards = [
                _mode_b_sme_reward(world.smes[p.sme_id], p, world.smes, coalition_held)
                for p in world.pairs.values()
                if p.sme_id in world.smes
            ]
            return _clamp01(sum(rewards) / max(len(rewards), 1)) if rewards else last_step_reward

        elif world.mode == WorldMode.C_OVERSIGHT:
            return last_step_reward  # accumulated per step

        elif world.mode == WorldMode.D_MANAGER:
            return _mode_d_manager_reward(world)

        return last_step_reward

    # ------------------------------------------- observation building

    def _build_observation(self, *, reward: float, done: bool) -> WorldObservation:
        world = self._world
        if world is None:
            return WorldObservation(
                world_step=0,
                episode_id="",
                mode="unknown",
                role="none",
                acting_agent_id="",
                text="Environment not initialized. Call reset() first.",
                reward=_STRICT_EPS,
                done=True,
                aux={},
                metadata={},
            )

        # Determine who this observation is for
        if self._schedule:
            role, agent_id = self._schedule[self._schedule_index % len(self._schedule)]
        else:
            role = "sme"
            agent_id = next(iter(world.smes), "sme_0")

        text = self._build_text_for_role(role, agent_id, world)

        aux: Dict[str, Any] = dict(world.metrics)
        aux["mode"] = world.mode.value
        aux["world_step"] = world.world_step
        aux["num_smes"] = len(world.smes)
        aux["num_buyers"] = len(world.buyers)
        aux["pairs_done"] = sum(1 for p in world.pairs.values() if p.done)
        aux["pairs_total"] = len(world.pairs)

        # Add per-SME metrics for logging
        sme_metrics: Dict[str, Any] = {}
        for sme_id, sme in world.smes.items():
            sme_metrics[sme_id] = {
                "is_solvent": sme.is_solvent,
                "current_payment_days": sme.current_payment_days,
                "deal_done": sme.deal_done,
                "total_reward": sme.total_reward,
            }
        aux["sme_metrics"] = sme_metrics

        # Oversight-specific
        if world.mode == WorldMode.C_OVERSIGHT:
            ground_truth = _detect_ground_truth_unfair(world)
            aux["ground_truth_unfair_pairs"] = list(ground_truth)
            aux["oversight_precision_recall"] = _compute_precision_recall(
                world.oversight_flags, ground_truth
            )

        metadata: Dict[str, Any] = {
            "episode_id": world.episode_id,
            "task_name": world.mode.value,
            "termination_reason": world.termination_reason,
        }

        return WorldObservation(
            world_step=world.world_step,
            episode_id=world.episode_id,
            mode=world.mode.value,
            role=role,
            acting_agent_id=agent_id,
            text=text,
            reward=_clamp01(reward),
            done=done,
            aux=aux,
            metadata=metadata,
        )

    def _build_text_for_role(self, role: str, agent_id: str, world: WorldState) -> str:
        if role == "sme":
            sme = world.smes.get(agent_id)
            if sme is None:
                return f"[SME {agent_id}] Not found in world."

            pair = next((p for p in world.pairs.values() if p.sme_id == agent_id), None)
            if pair is None:
                return f"[SME {agent_id}] No active negotiation pair."

            buyer = world.buyers.get(pair.buyer_id)
            if buyer is None:
                return f"[SME {agent_id}] Buyer not found."

            competitors = [
                _sme_public_view(s)
                for sid, s in world.smes.items()
                if sid != agent_id
            ]
            coalition_text = world.coalition.as_text() if world.coalition else None
            manager_instr = world.manager_instructions.get(agent_id)

            return _build_sme_observation_text(
                sme,
                buyer,
                pair,
                world,
                competitor_public_views=competitors if world.mode == WorldMode.A_COMPETITIVE else None,
                coalition_text=coalition_text,
                manager_instruction=manager_instr,
            )

        elif role == "oversight":
            summaries = []
            for pair in world.pairs.values():
                sme = world.smes.get(pair.sme_id)
                buyer = world.buyers.get(pair.buyer_id)
                if sme is None or buyer is None:
                    continue
                compressed = pair.compress_history_for_oversight()
                last_buyer_days = compressed[-1].get("buyer_days", 90) if compressed else 90
                summaries.append({
                    "pair_id": pair.pair_id,
                    "buyer_days": last_buyer_days,
                    "buyer_power": buyer.power_score,
                    "sme_solvent": sme.is_solvent,
                    "liquidity_gap": max(0, last_buyer_days - sme.liquidity_threshold_days),
                    "round": pair.round_count,
                })
            return _build_oversight_observation_text(world, summaries)

        elif role == "manager":
            vol = sum(
                (p.final_price or 0) * world.buyers.get(p.buyer_id, BuyerProfile("", 0, 0, "", "", 0, 0)).contract_volume
                for p in world.pairs.values()
                if p.deal_reached and p.final_price is not None
            )
            return _build_manager_observation_text(world, {"total_volume": vol})

        return f"[{role}/{agent_id}] No observation available."


# ---------------------------------------------------------------------------
# Manager instruction parser
# ---------------------------------------------------------------------------


def _parse_manager_instruction_to_action(
    instruction: str,
    sme: SMEProfile,
    pair: NegotiationPair,
) -> Dict[str, Any]:
    """Parse a free-text manager instruction into a NegotiationAction dict.

    Supports keywords: 'accept', 'reject', 'treds', 'days=N', 'price=N'.
    Falls back to a safe heuristic propose if nothing is parsed.
    """
    if not instruction:
        return {
            "action_type": "propose",
            "price": round(sme.unit_cost * 1.15, 2),
            "payment_days": sme.liquidity_threshold_days,
            "use_treds": False,
        }

    lower = instruction.lower()
    action_type = "propose"
    use_treds = "treds" in lower

    if "accept" in lower:
        action_type = "accept"
        last_buyer = pair.turn_history[-1] if pair.turn_history else {}
        return {
            "action_type": "accept",
            "price": float(last_buyer.get("buyer_price", sme.unit_cost * 1.1)),
            "payment_days": int(last_buyer.get("buyer_days", sme.liquidity_threshold_days)),
            "use_treds": use_treds,
            "reason": f"Manager instructed: {instruction[:80]}",
        }

    if "reject" in lower:
        return {
            "action_type": "reject",
            "price": round(sme.unit_cost * 1.15, 2),
            "payment_days": sme.liquidity_threshold_days,
            "use_treds": False,
            "reason": f"Manager instructed rejection: {instruction[:80]}",
        }

    # Parse numeric hints
    import re
    days_match = re.search(r"days[=\s]+(\d+)", lower)
    price_match = re.search(r"price[=\s]+([\d]+(?:\.[\d]+)?)", lower)

    target_days = int(days_match.group(1)) if days_match else sme.liquidity_threshold_days
    target_price = float(price_match.group(1)) if price_match else round(sme.unit_cost * 1.15, 2)

    target_price = max(sme.unit_cost * 1.01, target_price)

    return {
        "action_type": action_type,
        "price": round(target_price, 2),
        "payment_days": target_days,
        "use_treds": use_treds,
        "reason": f"Manager instructed: {instruction[:80]}",
    }


# ---------------------------------------------------------------------------
# Oversight metric helpers
# ---------------------------------------------------------------------------


def _compute_precision_recall(
    flagged: List[str],
    ground_truth: set,
) -> Dict[str, float]:
    flagged_set = set(flagged)
    tp = len(flagged_set & ground_truth)
    fp = len(flagged_set - ground_truth)
    fn = len(ground_truth - flagged_set)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1}
