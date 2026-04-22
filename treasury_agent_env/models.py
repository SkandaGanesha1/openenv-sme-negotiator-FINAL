"""Pydantic models for TreasuryAgent: Action, Observation, State."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import ConfigDict, Field
from openenv.core import Action, Observation, State


class TreasuryAction(Action):
    """
    One structured JSON tool call from the treasury agent.
    Maps exactly to the system-prompt tool interface.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    role: Literal["treasury"] = "treasury"
    command_type: Literal["tool_call", "policy_action"] = "tool_call"
    app: Literal[
        "erp_app",
        "bank_app",
        "treds_app",
        "dd_app",
        "compliance_app",
        "analytics_app",
    ]
    endpoint: str = Field(..., min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class TreasuryObservation(Observation):
    """
    Partial observation returned to the agent after each step.
    Omits hidden world state (POMDP design: buyer credit scores,
    true vendor stress, upcoming invoice amounts are not disclosed).
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Identity / time
    primary_sme_id: str
    day: int
    max_days: int
    step_count: int
    max_steps: int
    difficulty: str
    task_name: str

    # Observable KPIs
    cash_buffer_days: float = Field(
        ..., description="Days of operating runway at current burn rate"
    )
    dso_days: float = Field(
        ..., description="Days Sales Outstanding across all buyers"
    )
    vendor_stress_score: float = Field(
        ..., ge=0.0, le=1.0, description="0=no stress, 1=maximum stress"
    )
    concentration_risk_hhi: float = Field(
        ..., ge=0.0, le=1.0, description="Herfindahl-Hirschman Index of buyer revenue"
    )

    # Balance sheet summary
    overdraft_used: float
    overdraft_limit: float
    pending_invoice_count: int
    total_receivables: float
    upcoming_payables_30d: float
    treds_eligible_amount: float
    compliance_breach_count: int

    # Tool feedback (POMDP: tool results are the primary information channel)
    last_tool_result: dict[str, Any] = Field(default_factory=dict)

    # Episode signals
    solvency_ok: bool = True
    step_reward: float = 0.0
    cumulative_reward: float = 0.0
    message: str = ""


class TreasuryState(State):
    """
    Full episode state — includes hidden world snapshot.
    Used by graders only; never serialized into the agent's observation.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    episode_id: str
    seed: int
    difficulty: str
    task_name: str

    day: int
    max_days: int
    step_count: int
    max_steps: int

    solvency_breached: bool = False
    solvency_breach_day: Optional[int] = None
    total_financing_cost: float = 0.0
    total_revenue_collected: float = 0.0
    total_vendor_overdue_days: float = 0.0
    actions_taken: list[dict[str, Any]] = Field(default_factory=list)

    # Grader-visible snapshot (populated by environment on each step)
    world_snapshot: dict[str, Any] = Field(default_factory=dict)
