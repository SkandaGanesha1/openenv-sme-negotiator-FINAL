"""
Pydantic models for TreasuryCommandCenter environment.

Design principles:
- TreasuryCCAction: unified action covering all 5 modes + 6 tool apps + multi-agent fields
- TreasuryCCObservation: POMDP partial observation with GRPO group signals + world-model entropy
- TreasuryCCState: full hidden state for graders only (never serialized to agent)
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import ConfigDict, Field
from openenv.core import Action, Observation, State


# ── Modes ─────────────────────────────────────────────────────────────────────

TreasuryCCModeStr = Literal[
    "treasury-solo",        # single Treasury AI Officer manages all SMEs via tool calls
    "treasury-multi",       # per-SME agent with real tool access; CTDE scheduling
    "treasury-coalition",   # SMEs post coalition messages before tool calls
    "treasury-oversight",   # OversightAgent reads analytics dashboard + flags risks
    "treasury-manager",     # ManagerAgent issues per-SME instructions
]

RoleStr = Literal[
    "treasury_officer",     # SOLO: omniscient treasury officer
    "sme_agent",            # MULTI/COALITION: individual SME agent
    "oversight_agent",      # OVERSIGHT: reads compressed summaries, flags risks
    "manager_agent",        # MANAGER: issues instructions, queries analytics
]

AppStr = Literal[
    "erp_app",
    "bank_app",
    "treds_app",
    "dd_app",
    "compliance_app",
    "analytics_app",
]


# ── Action ────────────────────────────────────────────────────────────────────

class TreasuryCCAction(Action):
    """
    Unified action for all 5 TreasuryCommandCenter modes.

    Role routing:
    - treasury_officer / sme_agent: uses app + endpoint + params + target_sme_id
    - oversight_agent: uses flag_risky_smes + suggested_interventions
    - manager_agent:   uses instructions + query_analytics

    GRPO training support:
    - group_id: set to the same string for N parallel rollouts on the same state
      → the reward shaper computes normalised advantages within the group
    """

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    role: RoleStr = Field("treasury_officer", description="Active agent role")
    acting_sme_id: Optional[str] = Field(
        None, description="SME this action targets (MULTI/COALITION modes)"
    )

    # ── Treasury tool call fields (treasury_officer / sme_agent) ──────────────
    app: Optional[AppStr] = Field(None, description="Tool app name")
    endpoint: Optional[str] = Field(None, description="App endpoint")
    params: Dict[str, Any] = Field(default_factory=dict, description="Endpoint kwargs")

    # ── Coalition field (COALITION mode) ──────────────────────────────────────
    coalition_message: Optional[str] = Field(
        None, description="Message posted to coalition channel before tool call"
    )

    # ── Oversight fields (OVERSIGHT mode) ─────────────────────────────────────
    flag_risky_smes: List[str] = Field(
        default_factory=list, description="SME IDs the oversight agent flags as at-risk"
    )
    suggested_interventions: Dict[str, str] = Field(
        default_factory=dict, description="Recommended action per flagged SME ID"
    )
    global_risk_summary: str = Field(
        "", description="Free-text pattern explanation from oversight agent"
    )

    # ── Manager fields (MANAGER mode) ─────────────────────────────────────────
    instructions: Dict[str, str] = Field(
        default_factory=dict, description="sme_id → instruction text"
    )
    query_analytics: Optional[str] = Field(
        None,
        description="Analytics endpoint to query on behalf of manager: "
                    "portfolio_risks | scenario_analysis | kpi_dashboard | null",
    )

    # ── GRPO training fields ───────────────────────────────────────────────────
    group_id: Optional[str] = Field(
        None,
        description="Group identifier for GRPO. Set identical across N parallel "
                    "rollouts on the same observation to enable group reward normalisation.",
    )
    reasoning_trace: Optional[str] = Field(
        None,
        description="Chain-of-thought reasoning text from the LLM policy. "
                    "Stored in state for self-rewarding / LLM-as-judge scoring.",
    )


# ── Observation ───────────────────────────────────────────────────────────────

class TreasuryCCObservation(Observation):
    """
    Partial observation returned after each step (POMDP design).

    Hidden from agent:
    - Exact buyer credit scores & default probabilities
    - True vendor stress levels
    - Upcoming invoice amounts and exact arrival timing
    - Other SMEs' private cash / cost data (in MULTI/COALITION modes)

    Exposed GRPO signals:
    - step_reward / cumulative_reward for standard RL training
    - grpo_group_rewards: list of sibling-rollout rewards (populated when group_id set)
    - normalized_advantage: (step_reward - group_mean) / (group_std + eps)

    Exposed world-model signals:
    - belief_entropy: uncertainty over hidden world state (bits)
    - predicted_cash_buffer_30d: latent world model's 30-day cash forecast
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Identity / time
    mode: TreasuryCCModeStr
    role: RoleStr
    acting_sme_id: Optional[str]
    world_step: int
    day: int
    max_days: int
    step_count: int
    max_steps: int
    difficulty: str
    task_name: str
    episode_id: str

    # Observable KPIs (primary SME only — partial observability)
    cash_buffer_days: float = Field(..., description="Days of operating runway")
    dso_days: float = Field(..., description="Days Sales Outstanding")
    vendor_stress_score: float = Field(..., ge=0.0, le=1.0)
    concentration_risk_hhi: float = Field(..., ge=0.0, le=1.0)
    overdraft_used: float
    overdraft_limit: float
    pending_invoice_count: int
    total_receivables: float
    upcoming_payables_30d: float
    treds_eligible_amount: float
    compliance_breach_count: int
    solvency_ok: bool

    # Language-centric observation text (role-specific)
    text: str = Field(..., description="Rich language observation for LLM policy")

    # Tool feedback
    last_tool_result: Dict[str, Any] = Field(default_factory=dict)

    # Multi-agent context (MULTI / COALITION / OVERSIGHT / MANAGER)
    num_active_smes: int = 1
    coalition_channel_text: Optional[str] = None

    # GRPO training signals
    step_reward: float = 0.0
    cumulative_reward: float = 0.0
    grpo_group_rewards: List[float] = Field(
        default_factory=list,
        description="Rewards from sibling rollouts in same GRPO group",
    )
    normalized_advantage: float = Field(
        0.0, description="GRPO normalised advantage: (r - μ_group) / (σ_group + ε)"
    )

    # World-model signals (DreamerV3-inspired)
    belief_entropy: float = Field(
        0.0, description="Bits of uncertainty in the latent world model"
    )
    predicted_cash_buffer_30d: float = Field(
        0.0, description="Latent world model's 30-day cash buffer forecast"
    )
    world_model_prediction_error: float = Field(
        0.0,
        description="MAE between last prediction and observed KPI (self-improvement signal)",
    )

    # Curriculum signals
    curriculum_difficulty: float = Field(
        0.5, description="AutoCurriculum difficulty target [0,1]"
    )
    scenario_complexity: int = Field(
        1, description="Number of simultaneous stress factors active"
    )

    # Oversight-specific (OVERSIGHT mode aux)
    ground_truth_risky_smes: List[str] = Field(default_factory=list)
    oversight_precision: float = 0.0
    oversight_recall: float = 0.0
    oversight_f1: float = 0.0

    # Aux metrics dict for logging
    aux: Dict[str, Any] = Field(default_factory=dict)


# ── State (grader-visible only) ────────────────────────────────────────────────

class TreasuryCCState(State):
    """
    Full hidden episode state — grader-only, never serialized to agent observation.

    Contains:
    - world_snapshot: full treasury world (all cash, invoices, financing costs)
    - multi_agent_history: per-step action/obs log for self-rewarding judge
    - grpo_groups: group_id → list of (step, reward) for GRPO normalisation
    - curriculum_stats: rolling performance for AutoCurriculum
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    episode_id: str
    seed: int
    mode: TreasuryCCModeStr
    difficulty: str
    task_name: str

    day: int
    max_days: int
    step_count: int
    max_steps: int

    # Treasury world snapshot (hidden)
    solvency_breached: bool = False
    solvency_breach_day: Optional[int] = None
    total_financing_cost: float = 0.0
    total_revenue_collected: float = 0.0
    world_snapshot: Dict[str, Any] = Field(default_factory=dict)

    # Multi-agent history (for LLM-as-judge self-rewarding)
    actions_taken: List[Dict[str, Any]] = Field(default_factory=list)
    reasoning_traces: List[str] = Field(default_factory=list)

    # GRPO group tracking
    # group_id → list of per-step rewards collected across rollouts
    grpo_groups: Dict[str, List[float]] = Field(default_factory=dict)

    # World model tracking
    world_model_predictions: List[Dict[str, float]] = Field(default_factory=list)
    world_model_errors: List[float] = Field(default_factory=list)

    # Curriculum tracking
    recent_episode_rewards: List[float] = Field(default_factory=list)
    curriculum_difficulty: float = 0.5
