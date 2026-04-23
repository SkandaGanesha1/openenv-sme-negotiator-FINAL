"""
Task configuration registry for TreasuryCommandCenter environment.

Extends TreasuryTaskConfig with multi-agent parameters:
- agent_mode: which TreasuryCCMode to activate
- num_agents: how many LLM agents participate (1 in SOLO, N in MULTI)
- coalition_channel_capacity: max messages in coalition channel
- oversight_compression_ratio: fraction of history shown to oversight agent
- curriculum_enabled: whether AutoCurriculum adjusts difficulty
- grpo_group_size: N parallel rollouts for GRPO normalisation

Each difficulty reuses the existing TreasuryTaskConfig financial parameters
(SMEs, buyers, vendors, stochastic invoice params) and adds the above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple

from treasury_agent_env.task_config import (
    TreasuryTaskConfig,
    SMEConfig,
    BuyerConfig,
    VendorConfig,
    TASK_REGISTRY as _BASE_REGISTRY,
)

from .models import TreasuryCCModeStr


@dataclass(frozen=True)
class TreasuryCCTaskConfig:
    """Extended task config for TreasuryCommandCenter."""

    # Base treasury config (reused as-is for SimPy world)
    base: TreasuryTaskConfig

    # Multi-agent settings
    mode: TreasuryCCModeStr
    num_agents: int                       # LLM-controlled agents in this config
    coalition_channel_capacity: int = 8   # max coalition messages
    oversight_compression_ratio: float = 0.3  # fraction of turns visible to oversight
    manager_horizon_months: int = 3        # Mode MANAGER episode horizon in months

    # GRPO / training settings
    grpo_group_size: int = 4              # parallel rollouts per GRPO group
    curriculum_enabled: bool = True
    curriculum_target_success_rate: float = 0.5  # PAIRED target: ~50% success

    # Self-play / self-improvement
    self_rewarding_enabled: bool = False   # LLM-as-judge self-scoring
    constitutional_rules: Tuple[str, ...] = field(
        default_factory=lambda: (
            "Never violate solvency: keep all SME cash ≥ 0.",
            "Always check compliance before using Samadhaan (hard only).",
            "Prefer TReDS over overdraft when financing cost is lower.",
            "Minimise buyer concentration risk (HHI < 0.25).",
        )
    )

    # Shortcut properties delegating to base
    @property
    def name(self) -> str:
        return self.base.name

    @property
    def difficulty(self) -> str:
        return self.base.difficulty

    @property
    def description(self) -> str:
        return self.base.description

    @property
    def max_days(self) -> int:
        return self.base.max_days

    @property
    def max_steps(self) -> int:
        return self.base.max_steps

    @property
    def smes(self):
        return self.base.smes

    @property
    def buyers(self):
        return self.base.buyers

    @property
    def vendors(self):
        return self.base.vendors


# ── SOLO configs (single Treasury Officer, all 3 difficulties) ─────────────────

_SOLO_EASY = TreasuryCCTaskConfig(
    base=_BASE_REGISTRY["treasury-easy"],
    mode="treasury-solo",
    num_agents=1,
    grpo_group_size=4,
    curriculum_enabled=True,
    self_rewarding_enabled=False,
)

_SOLO_MEDIUM = TreasuryCCTaskConfig(
    base=_BASE_REGISTRY["treasury-medium"],
    mode="treasury-solo",
    num_agents=1,
    grpo_group_size=4,
    curriculum_enabled=True,
    self_rewarding_enabled=False,
)

_SOLO_HARD = TreasuryCCTaskConfig(
    base=_BASE_REGISTRY["treasury-hard"],
    mode="treasury-solo",
    num_agents=1,
    grpo_group_size=4,
    curriculum_enabled=True,
    self_rewarding_enabled=True,
)

# ── MULTI configs (per-SME agents, CTDE scheduling) ───────────────────────────

_MULTI_MEDIUM = TreasuryCCTaskConfig(
    base=_BASE_REGISTRY["treasury-medium"],
    mode="treasury-multi",
    num_agents=2,     # 2 SMEs, each LLM-controlled
    grpo_group_size=4,
    curriculum_enabled=True,
    self_rewarding_enabled=False,
)

_MULTI_HARD = TreasuryCCTaskConfig(
    base=_BASE_REGISTRY["treasury-hard"],
    mode="treasury-multi",
    num_agents=3,     # 3 SMEs, each LLM-controlled
    grpo_group_size=4,
    curriculum_enabled=True,
    self_rewarding_enabled=True,
)

# ── COALITION configs ─────────────────────────────────────────────────────────

_COALITION_MEDIUM = TreasuryCCTaskConfig(
    base=_BASE_REGISTRY["treasury-medium"],
    mode="treasury-coalition",
    num_agents=2,
    coalition_channel_capacity=10,
    grpo_group_size=4,
    curriculum_enabled=True,
)

_COALITION_HARD = TreasuryCCTaskConfig(
    base=_BASE_REGISTRY["treasury-hard"],
    mode="treasury-coalition",
    num_agents=3,
    coalition_channel_capacity=12,
    grpo_group_size=4,
    curriculum_enabled=True,
    self_rewarding_enabled=True,
)

# ── OVERSIGHT configs ─────────────────────────────────────────────────────────

_OVERSIGHT_HARD = TreasuryCCTaskConfig(
    base=_BASE_REGISTRY["treasury-hard"],
    mode="treasury-oversight",
    num_agents=1,     # one OversightAgent
    oversight_compression_ratio=0.3,
    grpo_group_size=4,
    curriculum_enabled=True,
)

# ── MANAGER configs ───────────────────────────────────────────────────────────

_MANAGER_HARD = TreasuryCCTaskConfig(
    base=_BASE_REGISTRY["treasury-hard"],
    mode="treasury-manager",
    num_agents=1,     # one ManagerAgent
    manager_horizon_months=6,
    grpo_group_size=4,
    curriculum_enabled=True,
    self_rewarding_enabled=True,
)


# ── Registry ──────────────────────────────────────────────────────────────────

TCC_TASK_REGISTRY: dict[str, TreasuryCCTaskConfig] = {
    # SOLO
    "tcc-solo-easy":       _SOLO_EASY,
    "tcc-solo-medium":     _SOLO_MEDIUM,
    "tcc-solo-hard":       _SOLO_HARD,
    # MULTI
    "tcc-multi-medium":    _MULTI_MEDIUM,
    "tcc-multi-hard":      _MULTI_HARD,
    # COALITION
    "tcc-coalition-medium": _COALITION_MEDIUM,
    "tcc-coalition-hard":   _COALITION_HARD,
    # OVERSIGHT
    "tcc-oversight-hard":  _OVERSIGHT_HARD,
    # MANAGER
    "tcc-manager-hard":    _MANAGER_HARD,
}

TCC_DIFFICULTY_DEFAULT: dict[str, str] = {
    "EASY":   "tcc-solo-easy",
    "MEDIUM": "tcc-solo-medium",
    "HARD":   "tcc-solo-hard",
}

MODE_DEFAULT_TASK: dict[str, str] = {
    "treasury-solo":       "tcc-solo-medium",
    "treasury-multi":      "tcc-multi-medium",
    "treasury-coalition":  "tcc-coalition-medium",
    "treasury-oversight":  "tcc-oversight-hard",
    "treasury-manager":    "tcc-manager-hard",
}


def resolve_tcc_task_id(
    task_name: str | None,
    difficulty: str = "MEDIUM",
) -> str:
    if task_name and task_name in TCC_TASK_REGISTRY:
        return task_name
    # Check if it's a mode name
    if task_name and task_name in MODE_DEFAULT_TASK:
        return MODE_DEFAULT_TASK[task_name]
    return TCC_DIFFICULTY_DEFAULT.get(difficulty.upper(), "tcc-solo-medium")
