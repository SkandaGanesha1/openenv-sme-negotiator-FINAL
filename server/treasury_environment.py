"""
TreasuryAgentEnvironment — openenv-core Environment subclass.

Architecture:
  reset()  → build TreasuryWorldState (SimPy + NetworkX) + BeliefState
             → advance SimPy 1 day to seed initial invoice population
             → return TreasuryObservation (partial — POMDP)

  step()   → route TreasuryAction to simulated tool app
             → advance SimPy 1 day
             → update BeliefState from events
             → compute multi-objective step reward
             → check termination (max_days or solvency breach)
             → return TreasuryObservation with last_tool_result
"""

from __future__ import annotations

import math
import os
import sys
from datetime import datetime, timezone
from random import Random
from typing import Optional

from openenv.core import Environment

from treasury_agent_env.action_router import TreasuryActionRouter
from treasury_agent_env.belief_state import TreasuryBeliefState
from treasury_agent_env.graders import TASK_GRADERS
from treasury_agent_env.models import (
    TreasuryAction,
    TreasuryObservation,
    TreasuryState,
)
from treasury_agent_env.reward import MultiObjectiveRewardShaper, _strict_unit_interval
from treasury_agent_env.task_config import (
    TASK_REGISTRY,
    TreasuryTaskConfig,
    resolve_task_id,
)
from treasury_agent_env.world_state import TreasuryWorldState


class TreasuryAgentEnvironment(Environment):
    """OpenEnv environment for SME treasury & supply-chain finance (POMDP)."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._rng = Random()
        self._task_config: TreasuryTaskConfig = TASK_REGISTRY["treasury-medium"]
        self._seed: int = 1000

        self._world: Optional[TreasuryWorldState] = None
        self._belief: Optional[TreasuryBeliefState] = None
        self._router: Optional[TreasuryActionRouter] = None
        self._shaper = MultiObjectiveRewardShaper()

        self._state: Optional[TreasuryState] = None
        self._cumulative_reward: float = 0.0
        self._last_tool_result: dict = {}

    # ── Internal helpers ───────────────────────────────────────────────────

    @property
    def state(self) -> Optional[TreasuryState]:
        return self._state

    def get_state(self) -> Optional[TreasuryState]:
        return self._state

    def _now_utc_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _grader_fn(self):
        return TASK_GRADERS.get(
            self._task_config.grader_id,
            TASK_GRADERS["treasury_balanced"],
        )

    def _primary_sme_id(self) -> str:
        return self._task_config.smes[0].sme_id

    def _build_observation(
        self,
        step_reward: float,
        message: str,
        reward: float,
        done: bool,
        metadata: dict,
    ) -> TreasuryObservation:
        assert self._world is not None
        assert self._state is not None
        tc = self._task_config
        primary = self._primary_sme_id()
        kpis = self._world.compute_kpis(primary)

        return TreasuryObservation(
            primary_sme_id=primary,
            day=self._world._current_day,
            max_days=tc.max_days,
            step_count=self._state.step_count,
            max_steps=tc.max_steps,
            difficulty=tc.difficulty,
            task_name=tc.name,
            # KPIs (partial observation)
            cash_buffer_days=kpis["cash_buffer_days"],
            dso_days=kpis["dso_days"],
            vendor_stress_score=kpis["vendor_stress_score"],
            concentration_risk_hhi=kpis["concentration_risk_hhi"],
            overdraft_used=kpis["overdraft_used"],
            overdraft_limit=kpis["overdraft_limit"],
            pending_invoice_count=kpis["pending_invoice_count"],
            total_receivables=kpis["total_receivables"],
            upcoming_payables_30d=kpis["upcoming_payables_30d"],
            treds_eligible_amount=kpis["treds_eligible_amount"],
            compliance_breach_count=kpis["compliance_breach_count"],
            solvency_ok=kpis["solvency_ok"],
            # Tool feedback
            last_tool_result=self._last_tool_result,
            # Episode signals
            step_reward=step_reward,
            cumulative_reward=self._cumulative_reward,
            message=message,
            # openenv-core required fields
            reward=reward,
            done=done,
            metadata=metadata,
        )

    def _update_state_snapshot(self) -> None:
        """Sync world snapshot into state for grader access."""
        assert self._world is not None
        assert self._state is not None
        snap = self._world.snapshot()
        # Compute HHI for grader
        cfg = self._task_config
        buyer_revenue: dict[str, float] = {}
        for sme in cfg.smes:
            for buyer in cfg.buyers:
                if self._world._graph.has_edge(sme.sme_id, buyer.buyer_id):
                    vol = self._world._graph[sme.sme_id][buyer.buyer_id]["annual_volume"]
                    buyer_revenue[buyer.buyer_id] = buyer_revenue.get(buyer.buyer_id, 0.0) + vol
        total_vol = sum(buyer_revenue.values()) or 1.0
        snap["hhi"] = sum((v / total_vol) ** 2 for v in buyer_revenue.values())
        self._state.world_snapshot = snap
        self._state.solvency_breached = not snap["solvency_ok"]
        self._state.solvency_breach_day = snap.get("solvency_breach_day")
        self._state.total_financing_cost = snap["total_financing_cost"]
        self._state.total_revenue_collected = sum(
            snap.get("revenue_collected", {}).values()
        )

    def _debug_reward(self, branch: str, r: float) -> None:
        if os.getenv("REWARD_DEBUG", "0").strip() not in ("0", "false", "False", "no"):
            print(
                f"[TREASURY_REWARD_DEBUG] branch={branch} reward={r:.4f}",
                file=sys.stderr,
                flush=True,
            )

    # ── Environment API ────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        difficulty: str = "MEDIUM",
        **kwargs,
    ) -> TreasuryObservation:
        """Reset the environment for a new episode."""
        requested_task = kwargs.get("task_name") or kwargs.get("task")
        task_id = resolve_task_id(
            str(requested_task) if requested_task else None,
            difficulty=difficulty,
        )
        self._task_config = TASK_REGISTRY[task_id]
        tc = self._task_config

        self._seed = int(seed if seed is not None else 1000)
        self._rng = Random(self._seed)
        self._cumulative_reward = 0.0
        self._last_tool_result = {}

        # Build world simulation
        self._world = TreasuryWorldState(tc, self._rng)

        # Build POMDP belief state
        buyer_credit_scores = {b.buyer_id: b.credit_score for b in tc.buyers}
        self._belief = TreasuryBeliefState(
            buyer_ids=[b.buyer_id for b in tc.buyers],
            vendor_ids=[v.vendor_id for v in tc.vendors],
            buyer_credit_scores=buyer_credit_scores,
        )

        # Build action router (holds refs to world + belief)
        self._router = TreasuryActionRouter(self._world, self._belief, self._rng)

        # Seed world with initial invoices (advance 1 day)
        init_events = self._world.advance(1)
        self._belief.update(init_events)

        episode_id = str(
            kwargs.get("episode_id") or f"{tc.difficulty}_{self._seed}_{self._now_utc_iso()[:10]}"
        )

        self._state = TreasuryState(
            episode_id=episode_id,
            seed=self._seed,
            difficulty=tc.difficulty,
            task_name=tc.name,
            day=self._world._current_day,
            max_days=tc.max_days,
            step_count=0,
            max_steps=tc.max_steps,
            solvency_breached=False,
            total_financing_cost=0.0,
            total_revenue_collected=0.0,
            world_snapshot={},
        )
        self._update_state_snapshot()

        message = (
            f"TreasuryAgent episode reset. Task: {tc.name} ({tc.difficulty}). "
            f"{tc.description} | {tc.context_note} "
            f"| Episode {episode_id} @ {self._now_utc_iso()}"
        )
        self._last_tool_result = {
            "type": "reset",
            "smes": [s.sme_id for s in tc.smes],
            "buyers": [b.buyer_id for b in tc.buyers],
            "vendors": [v.vendor_id for v in tc.vendors],
            "initial_invoices_generated": len(
                [e for e in init_events if e["type"] == "invoice_created"]
            ),
        }

        return self._build_observation(
            step_reward=0.0,
            message=message,
            reward=0.0,
            done=False,
            metadata={
                "episode_id": episode_id,
                "seed": self._seed,
                "task_id": tc.name,
                "task_description": tc.description,
                "max_days": tc.max_days,
                "max_steps": tc.max_steps,
                "treds_available": tc.treds_available,
                "dd_available": tc.dd_available,
                "compliance_active": tc.compliance_active,
            },
        )

    def step(self, action: TreasuryAction, **kwargs) -> TreasuryObservation:
        """Execute one tool call and advance the world by one day."""
        if self._state is None or self._world is None:
            self.reset(seed=self._seed)

        assert self._state is not None
        assert self._world is not None
        assert self._router is not None
        assert self._belief is not None

        tc = self._task_config

        # Already finished guard
        if (
            self._state.solvency_breached
            or self._state.step_count >= tc.max_steps
            or self._world._current_day >= tc.max_days
        ):
            self._update_state_snapshot()
            terminal = _strict_unit_interval(self._grader_fn()(self._state))
            return self._build_observation(
                step_reward=terminal,
                message="Episode already completed.",
                reward=terminal,
                done=True,
                metadata={"termination_reason": "already_completed"},
            )

        # ── Execute tool call ──────────────────────────────────────────────
        tool_result = self._router.route(action)
        self._last_tool_result = tool_result
        is_read_only = self._router.is_observational(action)

        # ── Advance world clock by 1 day ───────────────────────────────────
        events = self._world.advance(1)

        # ── Update belief state ────────────────────────────────────────────
        self._belief.update(events)

        # ── Advance step counter ───────────────────────────────────────────
        self._state.step_count += 1
        self._state.day = self._world._current_day
        self._state.actions_taken.append(
            {
                "step": self._state.step_count,
                "app": action.app,
                "endpoint": action.endpoint,
                "day": self._world._current_day,
            }
        )

        # ── Sync state snapshot ────────────────────────────────────────────
        self._update_state_snapshot()

        # ── Check termination ──────────────────────────────────────────────
        solvency_breached = self._state.solvency_breached
        max_steps_reached = self._state.step_count >= tc.max_steps
        max_days_reached = self._world._current_day >= tc.max_days
        done = solvency_breached or max_steps_reached or max_days_reached

        # ── Compute reward ─────────────────────────────────────────────────
        kpis = self._world.compute_kpis(self._primary_sme_id())
        # Include outstanding receivables so reward is live before payments arrive
        total_rev = self._state.total_revenue_collected + kpis["total_receivables"]

        if done:
            step_reward, components = self._shaper.terminal_reward(
                solvency_ok=not solvency_breached,
                solvency_breach_day=self._state.solvency_breach_day,
                max_days=tc.max_days,
                total_financing_cost=self._state.total_financing_cost,
                total_revenue=total_rev,
                vendor_stress_score=kpis["vendor_stress_score"],
                hhi=kpis["concentration_risk_hhi"],
            )
            step_reward = min(step_reward, 0.99)
            termination_reason = (
                "solvency_breach" if solvency_breached
                else "max_days_reached" if max_days_reached
                else "max_steps_reached"
            )
        else:
            step_reward, components = self._shaper.step_reward(
                solvency_ok=not solvency_breached,
                total_financing_cost=self._state.total_financing_cost,
                total_revenue=total_rev,
                vendor_stress_score=kpis["vendor_stress_score"],
                hhi=kpis["concentration_risk_hhi"],
                tool_was_useful=not is_read_only or "error" not in tool_result,
            )
            termination_reason = "ongoing"

        self._cumulative_reward += step_reward
        self._debug_reward(termination_reason, step_reward)

        message = (
            f"Day {self._world._current_day}/{tc.max_days} | "
            f"Step {self._state.step_count}/{tc.max_steps} | "
            f"Tool: {action.app}.{action.endpoint} | "
            f"Reward: {step_reward:.4f} | "
            f"{'DONE — ' + termination_reason if done else 'Ongoing'}"
        )

        return self._build_observation(
            step_reward=step_reward,
            message=message,
            reward=step_reward,
            done=done,
            metadata={
                "termination_reason": termination_reason,
                "reward_components": components,
                "solvency_ok": not solvency_breached,
                "day": self._world._current_day,
                "step": self._state.step_count,
            },
        )
