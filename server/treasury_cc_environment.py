"""
TreasuryCommandCenterEnvironment — OpenEnv POMDP environment.

Integrates:
  - TreasuryWorldState (SimPy + NetworkX hidden ground truth)
  - TreasuryBeliefState (Gaussian POMDP posteriors)
  - TreasuryActionRouter (6 real tool apps)
  - CTDEScheduler (multi-agent round-robin)
  - LatentWorldModel (DreamerV3-inspired RSSM)
  - GRPORewardShaper (RLVR + multi-objective)
  - AutoCurriculum (PAIRED self-play)

Five modes (task_name → mode):
  treasury-solo       → SOLO: single Treasury AI Officer, all tool apps
  treasury-multi      → MULTI: per-SME agents with CTDE scheduling
  treasury-coalition  → COALITION: SMEs coordinate via channel before tool calls
  treasury-oversight  → OVERSIGHT: OversightAgent reads analytics, flags risks
  treasury-manager    → MANAGER: ManagerAgent issues per-SME instructions

reset(**kwargs) → TreasuryCCObservation
step(TreasuryCCAction) → TreasuryCCObservation

GRPO training loop:
  1. reset() N times with same seed and group_id → N parallel episodes
  2. Each step() with same TreasuryCCAction but different rollout trajectories
  3. Collect grpo_group_rewards from obs.aux["grpo_group_rewards"]
  4. Compute normalized_advantage via GRPORewardShaper.normalise_grpo()
  5. Use advantage as policy gradient signal in TRL / Unsloth training
"""

from __future__ import annotations

import math
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from random import Random
from typing import Any, Dict, List, Optional, Tuple

from openenv.core import Environment

from treasury_agent_env.action_router import TreasuryActionRouter
from treasury_agent_env.belief_state import TreasuryBeliefState
from treasury_agent_env.world_state import TreasuryWorldState

from treasury_command_center_env.curriculum import AutoCurriculum, CurriculumParams
from treasury_command_center_env.graders import TCC_TASK_GRADERS, MODE_TO_GRADER
from treasury_command_center_env.models import (
    TreasuryCCAction,
    TreasuryCCObservation,
    TreasuryCCState,
)
from treasury_command_center_env.multi_agent import (
    AgentProfile,
    CoalitionChannel,
    CTDEScheduler,
    build_treasury_officer_text,
    build_sme_agent_text,
    build_oversight_text,
    build_manager_text,
)
from treasury_command_center_env.reward import (
    GRPORewardShaper,
    rubric_tool_quality,
    rubric_oversight_quality,
    rubric_manager_quality,
)
from treasury_command_center_env.task_config import (
    TreasuryCCTaskConfig,
    TCC_TASK_REGISTRY,
    resolve_tcc_task_id,
)
from treasury_command_center_env.world_model import LatentWorldModel


_EPS = 1e-6


def _strict_unit(x: float) -> float:
    if not math.isfinite(x):
        return _EPS
    return float(min(1.0 - _EPS, max(_EPS, x)))


def _gini(values: List[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    sv = sorted(max(v, 0.0) for v in values)
    total = sum(sv)
    if total < 1e-9:
        return 0.0
    gini_num = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(sv))
    return gini_num / (n * total)


class TreasuryCommandCenterEnvironment(Environment):
    """
    Treasury Command Center — combines multi-agent world scheduling with
    real SimPy treasury simulation, GRPO reward shaping, DreamerV3-inspired
    world model, and PAIRED autocurriculum.

    Compatible with OpenEnv HTTP + WebSocket APIs.
    Compatible with TRL and Unsloth GRPO training loops.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._rng = Random(42)
        self._task_config: Optional[TreasuryCCTaskConfig] = None
        self._seed: int = 42

        # Core simulation components (set at reset)
        self._world: Optional[TreasuryWorldState] = None
        self._belief: Optional[TreasuryBeliefState] = None
        self._router: Optional[TreasuryActionRouter] = None

        # Multi-agent components
        self._scheduler: Optional[CTDEScheduler] = None
        self._agents: Dict[str, AgentProfile] = {}
        self._coalition: Optional[CoalitionChannel] = None

        # RL components
        self._reward_shaper = GRPORewardShaper()
        self._world_model = LatentWorldModel()
        self._curriculum = AutoCurriculum()

        # Episode state
        self._state: Optional[TreasuryCCState] = None
        self._cumulative_reward: float = 0.0
        self._last_tool_result: Dict[str, Any] = {}
        self._world_step: int = 0
        self._month: int = 0

        # GRPO group tracking: group_id → list of step rewards
        self._grpo_groups: Dict[str, List[float]] = defaultdict(list)

        # Tool quality accumulation
        self._tool_quality_scores: List[float] = []

        # Constitution violation flag
        self._constitution_violated: bool = False

        # Oversight / Manager accumulation
        self._oversight_total_flags: List[str] = []
        self._oversight_interventions: Dict[str, str] = {}
        self._dso_at_start: float = 0.0

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def state(self) -> Optional[Dict[str, Any]]:
        if self._state is None:
            return None
        return {
            "episode_id": self._state.episode_id,
            "mode": self._state.mode,
            "world_step": self._world_step,
            "day": self._state.day,
            "solvency_breached": self._state.solvency_breached,
            "curriculum_difficulty": self._curriculum.difficulty,
        }

    def get_state(self) -> Optional[TreasuryCCState]:
        return self._state

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _primary_sme_id(self) -> str:
        assert self._task_config is not None
        return self._task_config.smes[0].sme_id

    def _all_sme_ids(self) -> List[str]:
        assert self._task_config is not None
        return [s.sme_id for s in self._task_config.smes]

    def _kpis_for(self, sme_id: str) -> Dict[str, Any]:
        assert self._world is not None
        return self._world.compute_kpis(sme_id)

    def _current_kpis(self) -> Dict[str, Any]:
        return self._kpis_for(self._primary_sme_id())

    def _detect_risky_smes(self) -> List[str]:
        """Ground-truth risky SMEs (used for oversight scoring)."""
        risky = []
        for sme_id in self._all_sme_ids():
            kpis = self._kpis_for(sme_id)
            if (
                not kpis.get("solvency_ok", True)
                or kpis.get("cash_buffer_days", 999) < 5.0
                or kpis.get("compliance_breach_count", 0) > 0
                or kpis.get("vendor_stress_score", 0) > 0.6
            ):
                risky.append(sme_id)
        return risky

    def _apply_curriculum_to_world(self, params: CurriculumParams) -> None:
        """
        Apply curriculum parameters to the world state after construction.
        Adjusts buyer payment delays and initial cash balances.
        """
        if self._world is None:
            return
        tc = self._task_config
        assert tc is not None

        # Scale working capital (cash) by curriculum stress factor
        stress = params.working_capital_stress
        for sme in tc.smes:
            sid = sme.sme_id
            if sid in self._world._cash:
                self._world._cash[sid] *= stress

        # Adjust overdraft limits (harder = less headroom)
        for sme in tc.smes:
            sid = sme.sme_id
            if sid in self._world._graph.nodes:
                od_limit = self._world._graph.nodes[sid].get("overdraft_limit", 0)
                self._world._graph.nodes[sid]["overdraft_limit"] = od_limit * stress

    def _compute_global_metrics(self) -> Dict[str, Any]:
        """Aggregate metrics across all SMEs for manager/oversight observations."""
        assert self._world is not None
        all_ids = self._all_sme_ids()
        solvent = sum(1 for sid in all_ids if self._kpis_for(sid).get("solvency_ok", True))
        avg_dso = sum(self._kpis_for(sid).get("dso_days", 0) for sid in all_ids) / max(len(all_ids), 1)
        total_fc = self._world.get_total_financing_cost()
        all_hhi = [self._kpis_for(sid).get("concentration_risk_hhi", 0) for sid in all_ids]
        portfolio_hhi = sum(all_hhi) / max(len(all_hhi), 1)
        return {
            "solvent_smes": solvent,
            "total_smes": len(all_ids),
            "avg_dso": round(avg_dso, 2),
            "total_financing_cost": round(total_fc, 2),
            "portfolio_hhi": round(portfolio_hhi, 4),
        }

    def _sme_compressed_summary(self, sme_id: str) -> Dict[str, Any]:
        """Compressed, partially-observable SME summary for oversight/manager."""
        kpis = self._kpis_for(sme_id)
        return {
            "sme_id": sme_id,
            "solvency_ok": kpis.get("solvency_ok", True),
            "cash_buffer_days": round(kpis.get("cash_buffer_days", 0), 1),
            "vendor_stress_score": round(kpis.get("vendor_stress_score", 0), 3),
            "compliance_breach_count": kpis.get("compliance_breach_count", 0),
            "treds_eligible_amount": round(kpis.get("treds_eligible_amount", 0), 0),
        }

    def _peer_sme_summaries(self, acting_sme_id: str) -> List[Dict[str, Any]]:
        """Public-only peer summaries (no private cost/cash data)."""
        return [
            self._sme_compressed_summary(sid)
            for sid in self._all_sme_ids()
            if sid != acting_sme_id
        ]

    def _build_snapshot_for_grader(self) -> Dict[str, Any]:
        assert self._world is not None
        assert self._state is not None
        snap = self._world.snapshot()
        tc = self._task_config
        assert tc is not None

        # Compute HHI
        buyer_revenue: Dict[str, float] = {}
        for sme in tc.smes:
            for buyer in tc.buyers:
                if self._world._graph.has_edge(sme.sme_id, buyer.buyer_id):
                    vol = self._world._graph[sme.sme_id][buyer.buyer_id]["annual_volume"]
                    buyer_revenue[buyer.buyer_id] = buyer_revenue.get(buyer.buyer_id, 0.0) + vol
        total_vol = sum(buyer_revenue.values()) or 1.0
        snap["hhi"] = sum((v / total_vol) ** 2 for v in buyer_revenue.values())
        snap["max_days"] = tc.max_days

        # Multi-agent extensions
        per_sme_snapshots = []
        for sme in tc.smes:
            sid = sme.sme_id
            kpis = self._kpis_for(sid)
            per_sme_snapshots.append({
                **snap,
                "sme_id": sid,
                "cash": {sid: self._world._cash.get(sid, 0)},
                "revenue_collected": {sid: self._world._revenue_collected.get(sid, 0)},
                "solvency_ok": kpis.get("solvency_ok", snap.get("solvency_ok", True)),
                "total_financing_cost": self._world.get_total_financing_cost(),
            })
        snap["per_sme_snapshots"] = per_sme_snapshots
        snap["total_smes"] = len(tc.smes)
        snap["solvent_smes"] = sum(1 for s in per_sme_snapshots if s.get("solvency_ok", True))

        # Gini on payment days (proxy via DSO)
        dso_vals = [self._kpis_for(sid).get("dso_days", 0) for sid in self._all_sme_ids()]
        snap["gini_payment_days"] = _gini(dso_vals)
        snap["avg_dso_improvement_days"] = max(0.0, self._dso_at_start - sum(dso_vals) / max(len(dso_vals), 1))

        # Tool quality
        snap["avg_tool_quality_score"] = (
            sum(self._tool_quality_scores) / len(self._tool_quality_scores)
            if self._tool_quality_scores else 0.5
        )
        snap["avg_world_model_error"] = self._world_model.avg_prediction_error()
        snap["constitution_violated"] = self._constitution_violated
        snap["compliance_breach_count"] = sum(
            self._kpis_for(sid).get("compliance_breach_count", 0)
            for sid in self._all_sme_ids()
        )

        # Oversight / manager specific
        snap["ground_truth_risky_smes"] = self._detect_risky_smes()
        snap["total_flagged_smes"] = self._oversight_total_flags
        snap["interventions"] = self._oversight_interventions
        snap["coalition_messages_posted"] = (
            len(self._coalition.messages) if self._coalition else 0
        )

        return snap

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        difficulty: str = "MEDIUM",
        **kwargs,
    ) -> TreasuryCCObservation:
        task_id = resolve_tcc_task_id(
            kwargs.get("task_name") or kwargs.get("task"),
            difficulty=difficulty,
        )
        self._task_config = TCC_TASK_REGISTRY[task_id]
        tc = self._task_config

        self._seed = int(seed if seed is not None else 42)
        self._rng = Random(self._seed)
        self._cumulative_reward = 0.0
        self._last_tool_result = {}
        self._world_step = 0
        self._month = 0
        self._grpo_groups = defaultdict(list)
        self._tool_quality_scores = []
        self._constitution_violated = False
        self._oversight_total_flags = []
        self._oversight_interventions = {}

        # Curriculum params
        curriculum_params = self._curriculum.sample_params(self._rng)

        # Build SimPy world
        self._world = TreasuryWorldState(tc.base, self._rng)

        # Apply curriculum stress to world
        if tc.curriculum_enabled:
            self._apply_curriculum_to_world(curriculum_params)

        # Build belief state
        buyer_credit_scores = {b.buyer_id: b.credit_score for b in tc.buyers}
        self._belief = TreasuryBeliefState(
            buyer_ids=[b.buyer_id for b in tc.buyers],
            vendor_ids=[v.vendor_id for v in tc.vendors],
            buyer_credit_scores=buyer_credit_scores,
        )

        # Build action router
        self._router = TreasuryActionRouter(self._world, self._belief, self._rng)

        # Seed world (advance 1 day)
        init_events = self._world.advance(1)
        self._belief.update(init_events)

        # Record initial DSO for manager reward
        self._dso_at_start = sum(
            self._kpis_for(sid).get("dso_days", 0) for sid in self._all_sme_ids()
        ) / max(len(self._all_sme_ids()), 1)

        # World model reset
        self._world_model = LatentWorldModel()
        primary_kpis = self._current_kpis()
        self._world_model.observe({k: float(primary_kpis.get(k, 0)) for k in
                                   ["cash_buffer_days", "dso_days", "vendor_stress_score", "concentration_risk_hhi"]})

        # Build scheduler + agents
        sme_ids = self._all_sme_ids()
        mode = tc.mode

        if mode == "treasury-solo":
            self._scheduler = CTDEScheduler.for_solo(sme_ids[0])
            self._agents = {
                "treasury_officer": AgentProfile("treasury_officer", "treasury_officer", sme_ids[0])
            }
            self._coalition = None

        elif mode == "treasury-multi":
            self._scheduler = CTDEScheduler.for_multi(sme_ids)
            self._agents = {
                sid: AgentProfile(sid, "sme_agent", sid) for sid in sme_ids
            }
            self._coalition = None

        elif mode == "treasury-coalition":
            self._scheduler = CTDEScheduler.for_coalition(sme_ids)
            self._agents = {
                sid: AgentProfile(sid, "sme_agent", sid) for sid in sme_ids
            }
            self._coalition = CoalitionChannel(max_messages=tc.coalition_channel_capacity)

        elif mode == "treasury-oversight":
            self._scheduler = CTDEScheduler.for_oversight()
            self._agents = {
                "oversight_0": AgentProfile("oversight_0", "oversight_agent", None)
            }
            self._coalition = None

        elif mode == "treasury-manager":
            self._scheduler = CTDEScheduler.for_manager()
            self._agents = {
                "manager_0": AgentProfile("manager_0", "manager_agent", None)
            }
            self._coalition = None

        else:
            self._scheduler = CTDEScheduler.for_solo(sme_ids[0])
            self._agents = {}
            self._coalition = None

        episode_id = str(kwargs.get("episode_id") or f"tcc_{mode}_{uuid.uuid4().hex[:8]}")

        self._state = TreasuryCCState(
            episode_id=episode_id,
            seed=self._seed,
            mode=mode,
            difficulty=tc.difficulty,
            task_name=tc.name,
            day=self._world._current_day,
            max_days=tc.max_days,
            step_count=0,
            max_steps=tc.max_steps,
            curriculum_difficulty=curriculum_params.difficulty,
        )

        self._last_tool_result = {
            "type": "reset",
            "task": tc.name,
            "mode": mode,
            "smes": sme_ids,
            "buyers": [b.buyer_id for b in tc.buyers],
            "vendors": [v.vendor_id for v in tc.vendors],
            "curriculum_difficulty": curriculum_params.difficulty,
            "scenario_complexity": curriculum_params.scenario_complexity,
        }

        return self._build_observation(step_reward=0.0, done=False)

    # ── Step ───────────────────────────────────────────────────────────────────

    def step(self, action: TreasuryCCAction, **kwargs) -> TreasuryCCObservation:
        if self._state is None or self._world is None:
            self.reset(seed=self._seed)

        assert self._state is not None
        assert self._world is not None
        assert self._router is not None
        assert self._belief is not None
        assert self._task_config is not None
        tc = self._task_config

        # Guard: already done
        already_done = (
            self._state.solvency_breached
            or self._state.step_count >= tc.max_steps
            or self._world._current_day >= tc.max_days
        )
        if already_done:
            snap = self._build_snapshot_for_grader()
            grader = TCC_TASK_GRADERS[MODE_TO_GRADER.get(tc.mode, "tcc_solo")]
            terminal = grader(snap)
            return self._build_observation(step_reward=_strict_unit(terminal), done=True)

        # ── Store reasoning trace ──────────────────────────────────────────────
        if action.reasoning_trace:
            self._state.reasoning_traces.append(action.reasoning_trace[:2000])

        # ── Constitutional check ───────────────────────────────────────────────
        primary_kpis = self._current_kpis()
        constitution_check = self._curriculum.check_constitutional_violation(
            action_app=action.app,
            action_endpoint=action.endpoint,
            kpis=primary_kpis,
            constitutional_rules=tc.base.__class__.__dict__.get("constitutional_rules", ()),
        )
        if constitution_check:
            self._constitution_violated = True

        # ── Route action to correct handler ───────────────────────────────────
        step_reward = 0.0
        role = action.role
        mode = tc.mode

        if role in ("treasury_officer", "sme_agent"):
            step_reward = self._step_tool_call(action, primary_kpis)
        elif role == "oversight_agent":
            step_reward = self._step_oversight(action)
        elif role == "manager_agent":
            step_reward = self._step_manager(action)
        else:
            step_reward = _EPS

        # ── Advance SimPy world by 1 day ───────────────────────────────────────
        events = self._world.advance(1)
        self._belief.update(events)

        # ── World model update ─────────────────────────────────────────────────
        new_kpis = self._current_kpis()
        self._world_model.observe({k: float(new_kpis.get(k, 0)) for k in
                                   ["cash_buffer_days", "dso_days", "vendor_stress_score", "concentration_risk_hhi"]})

        # ── Advance state counters ─────────────────────────────────────────────
        self._state.step_count += 1
        self._state.day = self._world._current_day
        self._world_step += 1
        self._scheduler.advance()

        # Update month (for manager mode)
        if mode == "treasury-manager":
            steps_per_month = max(1, len(self._all_sme_ids()))
            self._month = self._world_step // steps_per_month

        # ── Sync state snapshot ────────────────────────────────────────────────
        snap = self._world.snapshot()
        self._state.solvency_breached = not snap.get("solvency_ok", True)
        self._state.solvency_breach_day = snap.get("solvency_breach_day")
        self._state.total_financing_cost = snap.get("total_financing_cost", 0.0)
        self._state.total_revenue_collected = sum(snap.get("revenue_collected", {}).values())

        # ── Log action ─────────────────────────────────────────────────────────
        self._state.actions_taken.append({
            "step": self._state.step_count,
            "role": role,
            "app": action.app,
            "endpoint": action.endpoint,
            "day": self._world._current_day,
        })

        # ── GRPO group tracking ────────────────────────────────────────────────
        if action.group_id:
            self._grpo_groups[action.group_id].append(step_reward)
            if action.group_id not in self._state.grpo_groups:
                self._state.grpo_groups[action.group_id] = []
            self._state.grpo_groups[action.group_id].append(step_reward)

        # ── Check termination ──────────────────────────────────────────────────
        solvency_breached = self._state.solvency_breached
        max_steps_hit = self._state.step_count >= tc.max_steps
        max_days_hit = self._world._current_day >= tc.max_days
        manager_done = (
            mode == "treasury-manager"
            and self._month >= tc.manager_horizon_months
        )
        done = solvency_breached or max_steps_hit or max_days_hit or manager_done

        if done:
            step_reward = self._compute_terminal_reward()

        self._cumulative_reward += step_reward
        return self._build_observation(step_reward=step_reward, done=done)

    # ── Mode handlers ──────────────────────────────────────────────────────────

    def _step_tool_call(
        self, action: TreasuryCCAction, kpis: Dict[str, Any]
    ) -> float:
        """Handle treasury tool call (SOLO + MULTI + COALITION modes)."""
        assert self._router is not None
        assert self._state is not None

        # Coalition message (COALITION mode)
        if action.coalition_message and self._coalition is not None:
            acting_id = action.acting_sme_id or self._primary_sme_id()
            self._coalition.post(acting_id, action.coalition_message)

        # Execute tool call
        if action.app and action.endpoint:
            from treasury_agent_env.models import TreasuryAction
            try:
                tool_action = TreasuryAction(
                    role="treasury",
                    command_type="tool_call",
                    app=action.app,
                    endpoint=action.endpoint,
                    params=action.params or {},
                )
                result = self._router.route(tool_action)
                self._last_tool_result = result
                is_read_only = self._router.is_observational(tool_action)
            except Exception as exc:
                self._last_tool_result = {"error": str(exc), "app": action.app}
                is_read_only = True
        else:
            self._last_tool_result = {"type": "no_tool_call"}
            is_read_only = True

        # Tool quality rubric
        tq_score = rubric_tool_quality(
            app=action.app,
            endpoint=action.endpoint,
            kpis=kpis,
            step_in_episode=self._state.step_count,
        )
        self._tool_quality_scores.append(tq_score)

        # Step reward
        wm_error = self._world_model.prediction_error()
        sr, _ = self._reward_shaper.step_reward(
            solvency_ok=bool(kpis.get("solvency_ok", True)),
            total_financing_cost=self._state.total_financing_cost,
            total_revenue=max(self._state.total_revenue_collected + float(kpis.get("total_receivables", 0)), 1.0),
            vendor_stress_score=float(kpis.get("vendor_stress_score", 0)),
            hhi=float(kpis.get("concentration_risk_hhi", 0)),
            tool_quality_score=tq_score,
            world_model_error=wm_error,
            compliance_breach_count=int(kpis.get("compliance_breach_count", 0)),
            constitution_violated=self._constitution_violated,
        )
        return sr

    def _step_oversight(self, action: TreasuryCCAction) -> float:
        """Handle oversight agent action."""
        # Update cumulative oversight state
        for sid in action.flag_risky_smes:
            if sid not in self._oversight_total_flags:
                self._oversight_total_flags.append(sid)
        self._oversight_interventions.update(action.suggested_interventions)

        ground_truth = self._detect_risky_smes()
        reward, _ = self._reward_shaper.oversight_reward(
            flagged_smes=action.flag_risky_smes,
            ground_truth_risky=ground_truth,
            interventions=action.suggested_interventions,
        )
        return reward

    def _step_manager(self, action: TreasuryCCAction) -> float:
        """Handle manager agent action."""
        assert self._world is not None
        assert self._state is not None

        # Query analytics if requested
        if action.query_analytics and self._router:
            from treasury_agent_env.models import TreasuryAction
            try:
                analytics_map = {
                    "portfolio_risks": ("analytics_app", "portfolio_risks", {}),
                    "scenario_analysis": ("analytics_app", "scenario_analysis", {}),
                    "kpi_dashboard": ("analytics_app", "kpi_dashboard", {"sme_id": self._primary_sme_id()}),
                }
                if action.query_analytics in analytics_map:
                    app, ep, params = analytics_map[action.query_analytics]
                    tool_action = TreasuryAction(
                        role="treasury", command_type="tool_call",
                        app=app, endpoint=ep, params=params,
                    )
                    self._last_tool_result = self._router.route(tool_action)
            except Exception:
                pass

        # Identify SMEs needing help
        smes_needing_help = [
            sid for sid in self._all_sme_ids()
            if not self._kpis_for(sid).get("solvency_ok", True)
            or self._kpis_for(sid).get("cash_buffer_days", 999) < 10.0
        ]

        instr_quality = rubric_manager_quality(
            instructions=action.instructions,
            sme_ids_needing_help=smes_needing_help,
            query_analytics=action.query_analytics,
        )
        self._tool_quality_scores.append(instr_quality)

        # Apply instructions: try to route each to a simple analytics call on behalf of SME
        for sme_id, instruction in action.instructions.items():
            # Instructions are advisory in this mode — stored for self-rewarding judge
            if sme_id in self._all_sme_ids():
                self._state.actions_taken.append({
                    "step": self._state.step_count,
                    "role": "manager",
                    "instruction_to": sme_id,
                    "instruction": instruction[:200],
                })

        gm = self._compute_global_metrics()
        solvent_frac = gm["solvent_smes"] / max(gm["total_smes"], 1)
        dso_vals = [self._kpis_for(sid).get("dso_days", 0) for sid in self._all_sme_ids()]
        gini = _gini(dso_vals)

        reward, _ = self._reward_shaper.manager_reward(
            solvent_fraction=solvent_frac,
            avg_dso_improvement=max(0.0, self._dso_at_start - sum(dso_vals) / max(len(dso_vals), 1)),
            gini_days=gini,
            total_financing_cost=self._state.total_financing_cost,
            total_revenue=max(self._state.total_revenue_collected, 1.0),
            instruction_quality=instr_quality,
        )
        return reward

    def _compute_terminal_reward(self) -> float:
        assert self._task_config is not None
        snap = self._build_snapshot_for_grader()
        grader_id = MODE_TO_GRADER.get(self._task_config.mode, "tcc_solo")
        grader = TCC_TASK_GRADERS[grader_id]
        terminal = grader(snap)
        return _strict_unit(terminal)

    # ── Observation builder ────────────────────────────────────────────────────

    def _build_observation(self, *, step_reward: float, done: bool) -> TreasuryCCObservation:
        assert self._task_config is not None
        assert self._state is not None
        tc = self._task_config

        if self._world is None:
            return TreasuryCCObservation(
                mode=tc.mode,
                role="treasury_officer",
                acting_sme_id=None,
                world_step=0,
                day=0,
                max_days=tc.max_days,
                step_count=0,
                max_steps=tc.max_steps,
                difficulty=tc.difficulty,
                task_name=tc.name,
                episode_id="",
                cash_buffer_days=0.0,
                dso_days=0.0,
                vendor_stress_score=0.0,
                concentration_risk_hhi=0.0,
                overdraft_used=0.0,
                overdraft_limit=0.0,
                pending_invoice_count=0,
                total_receivables=0.0,
                upcoming_payables_30d=0.0,
                treds_eligible_amount=0.0,
                compliance_breach_count=0,
                solvency_ok=True,
                text="Environment not initialized.",
                reward=_EPS,
                done=True,
            )

        # Determine current role and acting agent
        role, acting_id = self._scheduler.current()
        sme_id = acting_id if role in ("treasury_officer", "sme_agent") else self._primary_sme_id()
        kpis = self._kpis_for(sme_id)

        # World model signals
        wm = self._world_model
        belief_entropy = wm.belief_entropy()
        predicted_cash = wm.cash_buffer_forecast_30d()
        wm_error = wm.prediction_error()

        # Build role-specific observation text
        text = self._build_text(role, sme_id, kpis)

        # GRPO group rewards
        active_group_id = getattr(self, "_last_group_id", None)
        group_rewards: List[float] = []
        norm_advantage = 0.0
        if active_group_id and active_group_id in self._grpo_groups:
            group_rewards = list(self._grpo_groups[active_group_id])
            if len(group_rewards) > 1:
                advantages = GRPORewardShaper.normalise_grpo(group_rewards)
                norm_advantage = advantages[-1] if advantages else 0.0

        # GRPO success probability (learning progress tracker)
        all_rewards = [r for g in self._grpo_groups.values() for r in g]
        grpo_success_p = GRPORewardShaper.grpo_success_probability_gain(all_rewards)

        # Oversight precision/recall (Mode OVERSIGHT)
        gt_risky = self._detect_risky_smes()
        if role == "oversight_agent" and self._oversight_total_flags:
            gt_set = set(gt_risky)
            flag_set = set(self._oversight_total_flags)
            tp = len(gt_set & flag_set)
            fp = len(flag_set - gt_set)
            fn = len(gt_set - flag_set)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, _EPS)
        else:
            prec = rec = f1 = 0.0

        aux: Dict[str, Any] = {
            "world_step": self._world_step,
            "mode": tc.mode,
            "curriculum": self._curriculum.state_dict(),
            "world_model": wm.summary(),
            "grpo_success_probability": round(grpo_success_p, 4),
            "global_metrics": self._compute_global_metrics(),
            "solvency_breached": self._state.solvency_breached,
        }
        if self._last_tool_result:
            aux["last_tool_result"] = self._last_tool_result

        # Store group_id from latest action for next obs
        return TreasuryCCObservation(
            mode=tc.mode,
            role=role,
            acting_sme_id=sme_id if role in ("treasury_officer", "sme_agent") else None,
            world_step=self._world_step,
            day=self._world._current_day,
            max_days=tc.max_days,
            step_count=self._state.step_count,
            max_steps=tc.max_steps,
            difficulty=tc.difficulty,
            task_name=tc.name,
            episode_id=self._state.episode_id,
            # KPIs
            cash_buffer_days=float(kpis.get("cash_buffer_days", 0.0)),
            dso_days=float(kpis.get("dso_days", 0.0)),
            vendor_stress_score=float(kpis.get("vendor_stress_score", 0.0)),
            concentration_risk_hhi=float(kpis.get("concentration_risk_hhi", 0.0)),
            overdraft_used=float(kpis.get("overdraft_used", 0.0)),
            overdraft_limit=float(kpis.get("overdraft_limit", 0.0)),
            pending_invoice_count=int(kpis.get("pending_invoice_count", 0)),
            total_receivables=float(kpis.get("total_receivables", 0.0)),
            upcoming_payables_30d=float(kpis.get("upcoming_payables_30d", 0.0)),
            treds_eligible_amount=float(kpis.get("treds_eligible_amount", 0.0)),
            compliance_breach_count=int(kpis.get("compliance_breach_count", 0)),
            solvency_ok=bool(kpis.get("solvency_ok", True)),
            # Text
            text=text,
            # Tool feedback
            last_tool_result=dict(self._last_tool_result),
            # Multi-agent
            num_active_smes=len(self._all_sme_ids()),
            coalition_channel_text=(
                self._coalition.as_text() if self._coalition else None
            ),
            # Rewards
            step_reward=_strict_unit(step_reward),
            cumulative_reward=_strict_unit(self._cumulative_reward + step_reward),
            grpo_group_rewards=group_rewards,
            normalized_advantage=round(norm_advantage, 6),
            # World model
            belief_entropy=belief_entropy,
            predicted_cash_buffer_30d=predicted_cash,
            world_model_prediction_error=wm_error,
            # Curriculum
            curriculum_difficulty=self._curriculum.difficulty,
            scenario_complexity=1,
            # Oversight
            ground_truth_risky_smes=gt_risky,
            oversight_precision=round(prec, 4),
            oversight_recall=round(rec, 4),
            oversight_f1=round(f1, 4),
            # OpenEnv required
            reward=_strict_unit(step_reward),
            done=done,
            metadata={
                "episode_id": self._state.episode_id,
                "task_name": tc.name,
                "mode": tc.mode,
                "day": self._world._current_day,
                "step": self._state.step_count,
            },
            aux=aux,
        )

    def _build_text(self, role: str, sme_id: str, kpis: Dict[str, Any]) -> str:
        assert self._task_config is not None
        tc = self._task_config
        wm = self._world_model

        if role == "treasury_officer":
            all_sme_summaries = [
                self._sme_compressed_summary(sid) for sid in self._all_sme_ids()
            ]
            return build_treasury_officer_text(
                kpis=kpis,
                all_sme_summaries=all_sme_summaries,
                world_step=self._world_step,
                day=self._world._current_day if self._world else 0,
                max_days=tc.max_days,
                last_tool_result=self._last_tool_result,
                belief_entropy=wm.belief_entropy(),
                predicted_cash_30d=wm.cash_buffer_forecast_30d(),
                curriculum_difficulty=self._curriculum.difficulty,
                constitutional_rules=tc.constitutional_rules,
            )

        elif role == "sme_agent":
            peers = self._peer_sme_summaries(sme_id)
            coalition_text = self._coalition.as_text() if self._coalition else None
            return build_sme_agent_text(
                sme_id=sme_id,
                kpis=kpis,
                world_step=self._world_step,
                day=self._world._current_day if self._world else 0,
                max_days=tc.max_days,
                last_tool_result=self._last_tool_result,
                coalition_text=coalition_text,
                peer_sme_summaries=peers,
                belief_entropy=wm.belief_entropy(),
                mode=tc.mode,
            )

        elif role == "oversight_agent":
            summaries = [self._sme_compressed_summary(sid) for sid in self._all_sme_ids()]
            return build_oversight_text(
                world_step=self._world_step,
                day=self._world._current_day if self._world else 0,
                sme_compressed_summaries=summaries,
                last_tool_result=self._last_tool_result,
            )

        elif role == "manager_agent":
            gm = self._compute_global_metrics()
            summaries = [self._sme_compressed_summary(sid) for sid in self._all_sme_ids()]
            return build_manager_text(
                world_step=self._world_step,
                day=self._world._current_day if self._world else 0,
                global_metrics=gm,
                sme_summaries=summaries,
                last_tool_result=self._last_tool_result,
                month=self._month,
                horizon_months=tc.manager_horizon_months,
            )

        return f"[{role}] No observation text available."
