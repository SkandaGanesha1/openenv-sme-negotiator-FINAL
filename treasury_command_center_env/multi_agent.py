"""
Multi-agent state management and CTDE scheduler for TreasuryCommandCenter.

Design:
  - AgentProfile: per-agent private state (role, SME assignment, private cost/cash known)
  - CoalitionChannel: shared SME message board (Mode COALITION)
  - CTDEScheduler: Centralised Training / Decentralised Execution round-robin
  - RoleObservationBuilder: builds role-specific, partial-observation text

Partial observability rules enforced here:
  - treasury_officer: sees all SME KPI aggregates (but not exact buyer credit scores)
  - sme_agent: sees ONLY its own SME's KPIs + public views of other SMEs
  - oversight_agent: sees compressed, sampled history (no private financials)
  - manager_agent: sees aggregate metrics + public SME summaries only
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── Agent profile ─────────────────────────────────────────────────────────────

@dataclass
class AgentProfile:
    """Private + public state tracked per LLM-controlled agent."""

    agent_id: str
    role: str                           # treasury_officer | sme_agent | ...
    assigned_sme_id: Optional[str]      # None for oversight/manager agents

    # Running performance
    total_reward: float = 0.0
    steps_taken: int = 0
    tool_calls: int = 0
    last_action_type: str = "none"

    # GRPO tracking
    group_id: Optional[str] = None
    group_rewards: List[float] = field(default_factory=list)

    # Self-rewarding / Constitutional AI tracking
    reasoning_traces: List[str] = field(default_factory=list)
    constitution_violations: int = 0


# ── Coalition channel ─────────────────────────────────────────────────────────

@dataclass
class CoalitionChannel:
    """
    Shared message board visible to all SME agents in COALITION mode.

    Analogous to CICERO's shared negotiation channel but for treasury coordination.
    Agents can post financing strategy proposals, warnings, or requests before
    executing their individual tool calls.
    """

    max_messages: int = 10
    messages: List[Dict[str, str]] = field(default_factory=list)

    def post(self, sender_id: str, text: str) -> None:
        if len(self.messages) < self.max_messages:
            self.messages.append({"sender": sender_id, "text": text[:500]})

    def as_text(self) -> str:
        if not self.messages:
            return "[coalition channel: empty]"
        return "\n".join(f"[{m['sender']}]: {m['text']}" for m in self.messages)

    def clear(self) -> None:
        self.messages.clear()


# ── CTDE Scheduler ────────────────────────────────────────────────────────────

class CTDEScheduler:
    """
    Centralised Training / Decentralised Execution scheduler.

    At training time:
    - The central trainer has access to the full TreasuryWorldState.
    - Each agent receives its own partial observation (role-specific).

    At execution time:
    - Agents act from their partial observation only.
    - The scheduler advances the round-robin and returns (role, sme_id).

    Modes:
    - SOLO:       schedule = [("treasury_officer", primary_sme_id)] repeating
    - MULTI:      schedule = [("sme_agent", sme_id) for each SME] round-robin
    - COALITION:  same as MULTI but with coalition channel available
    - OVERSIGHT:  schedule = [("oversight_agent", "oversight_0")]
    - MANAGER:    schedule = [("manager_agent", "manager_0")]
    """

    def __init__(self, schedule: List[Tuple[str, str]]) -> None:
        self._schedule = schedule
        self._idx = 0

    @classmethod
    def for_solo(cls, primary_sme_id: str) -> "CTDEScheduler":
        return cls([("treasury_officer", primary_sme_id)])

    @classmethod
    def for_multi(cls, sme_ids: List[str]) -> "CTDEScheduler":
        return cls([("sme_agent", sid) for sid in sme_ids])

    @classmethod
    def for_coalition(cls, sme_ids: List[str]) -> "CTDEScheduler":
        return cls([("sme_agent", sid) for sid in sme_ids])

    @classmethod
    def for_oversight(cls) -> "CTDEScheduler":
        return cls([("oversight_agent", "oversight_0")])

    @classmethod
    def for_manager(cls) -> "CTDEScheduler":
        return cls([("manager_agent", "manager_0")])

    def current(self) -> Tuple[str, str]:
        return self._schedule[self._idx % len(self._schedule)]

    def advance(self) -> None:
        self._idx += 1

    def peek_next(self) -> Tuple[str, str]:
        return self._schedule[(self._idx + 1) % len(self._schedule)]

    def reset(self) -> None:
        self._idx = 0


# ── Observation text builders (partial observability) ─────────────────────────

_ACTION_FORMAT_TREASURY = (
    "\nAction format (JSON):\n"
    '{"role": "treasury_officer", '
    '"app": "<erp_app|bank_app|treds_app|dd_app|compliance_app|analytics_app>", '
    '"endpoint": "<endpoint_name>", '
    '"params": {<endpoint_kwargs>}}'
)

_ACTION_FORMAT_SME = (
    "\nAction format (JSON):\n"
    '{"role": "sme_agent", "acting_sme_id": "<sme_id>", '
    '"app": "<erp_app|bank_app|treds_app|dd_app|compliance_app|analytics_app>", '
    '"endpoint": "<endpoint_name>", "params": {<kwargs>}, '
    '"coalition_message": "<optional message to coalition channel>"}'
)

_ACTION_FORMAT_OVERSIGHT = (
    "\nAction format (JSON):\n"
    '{"role": "oversight_agent", '
    '"flag_risky_smes": ["<sme_id>", ...], '
    '"suggested_interventions": {"<sme_id>": "<recommendation>"}, '
    '"global_risk_summary": "<free-text pattern summary>"}'
)

_ACTION_FORMAT_MANAGER = (
    "\nAction format (JSON):\n"
    '{"role": "manager_agent", '
    '"instructions": {"<sme_id>": "<instruction>"}, '
    '"query_analytics": "<portfolio_risks|scenario_analysis|kpi_dashboard|null>"}'
)

_TOOL_REFERENCE = """
Available tool apps:
  erp_app:        list_invoices, invoice_summary, projected_cashflow, update_terms
  bank_app:       get_balances, draw_overdraft, repay_overdraft, view_covenants
  treds_app:      quote_discount_rate, discount_invoice, eligibility_summary
  dd_app:         propose_discount_scheme, simulate_scheme, activate_scheme
  compliance_app: check_45_day_breach, estimate_43B_tax_impact, prepare_samadhaan_case
  analytics_app:  portfolio_risks, scenario_analysis, kpi_dashboard
"""


def build_treasury_officer_text(
    kpis: Dict[str, Any],
    all_sme_summaries: List[Dict[str, Any]],
    world_step: int,
    day: int,
    max_days: int,
    last_tool_result: Dict[str, Any],
    belief_entropy: float,
    predicted_cash_30d: float,
    curriculum_difficulty: float,
    constitutional_rules: Tuple[str, ...] = (),
) -> str:
    parts: List[str] = [
        f"[TreasuryOfficer] World step {world_step} | Day {day}/{max_days}",
        f"Portfolio KPIs (primary SME):",
        f"  cash_buffer_days={kpis.get('cash_buffer_days', 0):.1f}d  "
        f"dso={kpis.get('dso_days', 0):.1f}d  "
        f"vendor_stress={kpis.get('vendor_stress_score', 0):.3f}  "
        f"hhi={kpis.get('concentration_risk_hhi', 0):.3f}",
        f"  overdraft_used=₹{kpis.get('overdraft_used', 0):,.0f}/"
        f"₹{kpis.get('overdraft_limit', 0):,.0f}  "
        f"receivables=₹{kpis.get('total_receivables', 0):,.0f}  "
        f"solvency={'OK' if kpis.get('solvency_ok', True) else 'BREACH'}",
        f"  compliance_breaches={kpis.get('compliance_breach_count', 0)}  "
        f"treds_eligible=₹{kpis.get('treds_eligible_amount', 0):,.0f}",
    ]

    if all_sme_summaries:
        parts.append("All SMEs (public view):")
        for s in all_sme_summaries:
            parts.append(
                f"  {s['sme_id']}: cash_buffer={s.get('cash_buffer_days', '?'):.0f}d  "
                f"solvency={'OK' if s.get('solvency_ok', True) else 'BREACH'}  "
                f"financing_cost=₹{s.get('total_financing_cost', 0):,.0f}"
            )

    parts.append(
        f"World model: belief_entropy={belief_entropy:.2f}bits  "
        f"predicted_cash_buffer_30d={predicted_cash_30d:.1f}d"
    )
    parts.append(f"Curriculum difficulty={curriculum_difficulty:.2f}")

    if last_tool_result and "error" not in last_tool_result:
        tool_str = str(last_tool_result)[:300]
        parts.append(f"Last tool result: {tool_str}")
    elif last_tool_result and "error" in last_tool_result:
        parts.append(f"Last tool ERROR: {last_tool_result['error']}")

    if constitutional_rules:
        parts.append("Constitutional rules:")
        for r in constitutional_rules:
            parts.append(f"  • {r}")

    parts.append(_TOOL_REFERENCE)
    parts.append(_ACTION_FORMAT_TREASURY)
    return "\n".join(parts)


def build_sme_agent_text(
    sme_id: str,
    kpis: Dict[str, Any],
    world_step: int,
    day: int,
    max_days: int,
    last_tool_result: Dict[str, Any],
    coalition_text: Optional[str],
    peer_sme_summaries: List[Dict[str, Any]],
    belief_entropy: float,
    mode: str,
) -> str:
    parts: List[str] = [
        f"[SMEAgent {sme_id}] World step {world_step} | Day {day}/{max_days} | Mode: {mode}",
        f"Your KPIs:",
        f"  cash_buffer_days={kpis.get('cash_buffer_days', 0):.1f}d  "
        f"dso={kpis.get('dso_days', 0):.1f}d  "
        f"vendor_stress={kpis.get('vendor_stress_score', 0):.3f}",
        f"  overdraft=₹{kpis.get('overdraft_used', 0):,.0f}/₹{kpis.get('overdraft_limit', 0):,.0f}  "
        f"treds_eligible=₹{kpis.get('treds_eligible_amount', 0):,.0f}  "
        f"solvency={'OK' if kpis.get('solvency_ok', True) else 'BREACH'}",
    ]

    if peer_sme_summaries:
        parts.append("Peer SMEs (public view only — no private financials):")
        for p in peer_sme_summaries:
            parts.append(
                f"  {p['sme_id']}: solvency={'OK' if p.get('solvency_ok', True) else 'BREACH'}  "
                f"compliance_breaches={p.get('compliance_breach_count', 0)}"
            )

    if coalition_text is not None:
        parts.append(f"Coalition channel:\n{coalition_text}")

    parts.append(f"World model belief_entropy={belief_entropy:.2f}bits")

    if last_tool_result and "error" not in last_tool_result:
        parts.append(f"Last tool result: {str(last_tool_result)[:250]}")

    parts.append(_TOOL_REFERENCE)
    parts.append(_ACTION_FORMAT_SME)
    return "\n".join(parts)


def build_oversight_text(
    world_step: int,
    day: int,
    sme_compressed_summaries: List[Dict[str, Any]],
    last_tool_result: Dict[str, Any],
) -> str:
    parts: List[str] = [
        f"[OversightAgent] World step {world_step} | Day {day}",
        "Compressed SME risk summaries (partial view — sampled history):",
    ]
    for s in sme_compressed_summaries:
        risk = "CRITICAL" if not s.get("solvency_ok", True) else (
            "HIGH" if s.get("compliance_breach_count", 0) > 2 else "NORMAL"
        )
        parts.append(
            f"  {s['sme_id']}: risk={risk}  "
            f"cash_buffer={s.get('cash_buffer_days', 0):.0f}d  "
            f"vendor_stress={s.get('vendor_stress_score', 0):.3f}  "
            f"compliance_breaches={s.get('compliance_breach_count', 0)}"
        )

    if last_tool_result:
        parts.append(f"Last analytics result: {str(last_tool_result)[:250]}")

    parts.append(_ACTION_FORMAT_OVERSIGHT)
    return "\n".join(parts)


def build_manager_text(
    world_step: int,
    day: int,
    global_metrics: Dict[str, Any],
    sme_summaries: List[Dict[str, Any]],
    last_tool_result: Dict[str, Any],
    month: int,
    horizon_months: int,
) -> str:
    parts: List[str] = [
        f"[ManagerAgent] Month {month}/{horizon_months} | World step {world_step} | Day {day}",
        f"Global metrics: "
        f"solvent_smes={global_metrics.get('solvent_smes', 0)}/{global_metrics.get('total_smes', 0)}  "
        f"avg_dso={global_metrics.get('avg_dso', 0):.1f}d  "
        f"total_financing_cost=₹{global_metrics.get('total_financing_cost', 0):,.0f}  "
        f"portfolio_hhi={global_metrics.get('portfolio_hhi', 0):.3f}",
        "Per-SME status (no private cost data):",
    ]
    for s in sme_summaries:
        parts.append(
            f"  {s['sme_id']}: solvency={'OK' if s.get('solvency_ok', True) else 'BREACH'}  "
            f"cash_buffer={s.get('cash_buffer_days', 0):.0f}d  "
            f"treds_eligible=₹{s.get('treds_eligible_amount', 0):,.0f}"
        )

    if last_tool_result:
        parts.append(f"Last analytics: {str(last_tool_result)[:250]}")

    parts.append(_ACTION_FORMAT_MANAGER)
    return "\n".join(parts)
