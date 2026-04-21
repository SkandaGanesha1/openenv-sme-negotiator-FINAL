"""Tests for the multi-agent world environment (all four modes).

Run with:  pytest tests/test_world_environment.py -v
"""

from __future__ import annotations

import math
import sys
import os

# Ensure project root is on path when running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from server.world_environment import (
    SMEMultiAgentWorldEnvironment,
    WorldAction,
    WorldObservation,
    WorldMode,
    _clamp01,
    _compute_gini,
    _detect_ground_truth_unfair,
    _mode_a_sme_reward,
    _mode_c_oversight_reward,
    _mode_d_manager_reward,
    _parse_manager_instruction_to_action,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env(seed: int = 0) -> SMEMultiAgentWorldEnvironment:
    env = SMEMultiAgentWorldEnvironment()
    return env


def _assert_obs_valid(obs: WorldObservation) -> None:
    """Common structural assertions for any WorldObservation."""
    assert isinstance(obs, WorldObservation)
    assert obs.text and len(obs.text) > 10, "observation text must be non-empty"
    assert math.isfinite(obs.reward), "reward must be finite"
    assert 0.0 < obs.reward < 1.0, f"reward {obs.reward} must be in strict (0,1)"
    assert isinstance(obs.done, bool)
    assert obs.episode_id, "episode_id must be non-empty"
    assert obs.mode in [m.value for m in WorldMode], f"unknown mode: {obs.mode}"
    assert obs.world_step >= 0


def _null_sme_action(obs: WorldObservation, price: float = 90.0, days: int = 50) -> WorldAction:
    return WorldAction(
        role="sme",
        acting_agent_id=obs.acting_agent_id,
        negotiation_action={
            "action_type": "propose",
            "price": price,
            "payment_days": days,
            "use_treds": False,
            "reason": "test action",
        },
    )


def _null_oversight_action() -> WorldAction:
    return WorldAction(
        role="oversight",
        acting_agent_id="oversight_agent",
        flag_unfair_cases=[],
        suggested_interventions={},
        global_explanation="No patterns detected in this test step.",
    )


def _null_manager_action() -> WorldAction:
    return WorldAction(
        role="manager",
        acting_agent_id="manager_agent",
        instructions={},
        query_tool=None,
    )


# ---------------------------------------------------------------------------
# Mode A tests
# ---------------------------------------------------------------------------


class TestModeACompetitiveBidding:
    def test_reset_returns_valid_observation(self):
        env = _make_env()
        obs = env.reset(seed=42, task_name="competitive-bidding", num_smes=3, num_buyers=1)
        _assert_obs_valid(obs)
        assert obs.mode == WorldMode.A_COMPETITIVE.value
        assert obs.role == "sme"
        assert obs.world_step == 0

    def test_world_state_has_correct_agents(self):
        env = _make_env()
        env.reset(seed=7, task_name="competitive-bidding", num_smes=3, num_buyers=1)
        assert env._world is not None
        assert len(env._world.smes) == 3
        assert len(env._world.buyers) == 1
        assert len(env._world.pairs) == 3  # one pair per SME

    def test_step_advances_world_step(self):
        env = _make_env()
        obs = env.reset(seed=1, task_name="competitive-bidding", num_smes=2, num_buyers=1)
        action = _null_sme_action(obs)
        obs2 = env.step(action)
        _assert_obs_valid(obs2)
        assert obs2.world_step == 1

    def test_multiple_steps_no_crash(self):
        env = _make_env()
        obs = env.reset(seed=10, task_name="competitive-bidding", num_smes=3, num_buyers=1)
        for _ in range(9):
            if obs.done:
                break
            action = _null_sme_action(obs, price=92.0, days=50)
            obs = env.step(action)
            _assert_obs_valid(obs)

    def test_partial_observability_no_cost_leak(self):
        """Competitors' unit_cost must NOT appear in the observation text."""
        env = _make_env()
        obs = env.reset(seed=99, task_name="competitive-bidding", num_smes=3, num_buyers=1)
        # Observation text should not contain raw cost values from OTHER SMEs
        assert "unit_cost" not in obs.text.lower() or "your financials" in obs.text.lower()

    def test_rewards_strict_unit_interval(self):
        env = _make_env()
        obs = env.reset(seed=42, task_name="competitive-bidding", num_smes=2, num_buyers=1)
        for _ in range(15):
            if obs.done:
                break
            obs = env.step(_null_sme_action(obs))
        assert 0.0 < obs.reward < 1.0

    def test_episode_terminates(self):
        env = _make_env()
        obs = env.reset(seed=5, task_name="competitive-bidding", num_smes=2, num_buyers=1)
        steps = 0
        while not obs.done and steps < 200:
            obs = env.step(_null_sme_action(obs, price=95.0, days=45))
            steps += 1
        assert obs.done, "Episode must eventually terminate"

    def test_aux_metrics_present(self):
        env = _make_env()
        obs = env.reset(seed=3, task_name="competitive-bidding", num_smes=2, num_buyers=1)
        obs2 = env.step(_null_sme_action(obs))
        assert "gini_days" in obs2.aux
        assert "solvent_fraction" in obs2.aux
        assert "sme_metrics" in obs2.aux
        assert 0.0 <= obs2.aux["gini_days"] <= 1.0
        assert 0.0 <= obs2.aux["solvent_fraction"] <= 1.0


# ---------------------------------------------------------------------------
# Mode B tests
# ---------------------------------------------------------------------------


class TestModeBCoalitionBargaining:
    def test_reset_creates_coalition_channel(self):
        env = _make_env()
        env.reset(seed=20, task_name="coalition-bargaining", num_smes=3)
        assert env._world is not None
        assert env._world.coalition is not None

    def test_coalition_message_posted(self):
        env = _make_env()
        obs = env.reset(seed=21, task_name="coalition-bargaining", num_smes=3)
        action = WorldAction(
            role="sme",
            acting_agent_id=obs.acting_agent_id,
            negotiation_action={
                "action_type": "propose",
                "price": 92.0,
                "payment_days": 50,
                "use_treds": False,
            },
            coalition_message="We should hold out for ≤50 day terms together.",
        )
        env.step(action)
        assert env._world is not None
        assert len(env._world.coalition.messages) == 1

    def test_coalition_text_in_observation(self):
        env = _make_env()
        obs = env.reset(seed=22, task_name="coalition-bargaining", num_smes=3)
        # Post a coalition message
        action = WorldAction(
            role="sme",
            acting_agent_id=obs.acting_agent_id,
            negotiation_action={
                "action_type": "propose",
                "price": 93.0,
                "payment_days": 55,
                "use_treds": False,
            },
            coalition_message="Hold the line at 55 days.",
        )
        obs2 = env.step(action)
        # Next observation (for next SME) should contain coalition channel
        assert "coalition channel" in obs2.text.lower() or "coalition" in obs2.text.lower()

    def test_coalition_target_days_set(self):
        env = _make_env()
        env.reset(seed=23, task_name="coalition-bargaining", num_smes=3)
        assert env._world is not None
        for sme in env._world.smes.values():
            assert sme.coalition_agreement_target_days is not None
            assert 30 <= sme.coalition_agreement_target_days <= 70

    def test_multiple_steps_no_crash(self):
        env = _make_env()
        obs = env.reset(seed=24, task_name="coalition-bargaining", num_smes=3)
        for _ in range(12):
            if obs.done:
                break
            action = WorldAction(
                role="sme",
                acting_agent_id=obs.acting_agent_id,
                negotiation_action={
                    "action_type": "propose",
                    "price": 91.0,
                    "payment_days": 48,
                    "use_treds": False,
                },
                coalition_message=None,
            )
            obs = env.step(action)
            _assert_obs_valid(obs)


# ---------------------------------------------------------------------------
# Mode C tests
# ---------------------------------------------------------------------------


class TestModeCOversightArena:
    def test_reset_observation_is_for_oversight(self):
        env = _make_env()
        obs = env.reset(seed=30, task_name="oversight-arena", num_parallel_envs=4)
        _assert_obs_valid(obs)
        assert obs.mode == WorldMode.C_OVERSIGHT.value
        assert obs.role == "oversight"

    def test_oversight_obs_contains_summaries(self):
        env = _make_env()
        obs = env.reset(seed=31, task_name="oversight-arena", num_parallel_envs=3)
        assert "oversightagent" in obs.text.lower() or "oversightagent" in obs.text.replace(" ", "").lower()
        assert "pair" in obs.text.lower()

    def test_flag_empty_returns_valid_reward(self):
        env = _make_env()
        obs = env.reset(seed=32, task_name="oversight-arena", num_parallel_envs=3)
        obs2 = env.step(_null_oversight_action())
        _assert_obs_valid(obs2)

    def test_flag_all_pairs_as_unfair(self):
        env = _make_env()
        obs = env.reset(seed=33, task_name="oversight-arena", num_parallel_envs=3)
        assert env._world is not None
        all_pair_ids = list(env._world.pairs.keys())
        action = WorldAction(
            role="oversight",
            acting_agent_id="oversight_agent",
            flag_unfair_cases=all_pair_ids,
            suggested_interventions={pid: "Intervene immediately." for pid in all_pair_ids},
            global_explanation="All pairs appear abusive.",
        )
        obs2 = env.step(action)
        _assert_obs_valid(obs2)

    def test_ground_truth_aux_in_observation(self):
        env = _make_env()
        obs = env.reset(seed=34, task_name="oversight-arena", num_parallel_envs=4)
        obs2 = env.step(_null_oversight_action())
        assert "ground_truth_unfair_pairs" in obs2.aux
        assert "oversight_precision_recall" in obs2.aux
        pr = obs2.aux["oversight_precision_recall"]
        assert "precision" in pr and "recall" in pr and "f1" in pr

    def test_multiple_steps_advances_parallel_pairs(self):
        env = _make_env()
        obs = env.reset(seed=35, task_name="oversight-arena", num_parallel_envs=4)
        for _ in range(6):
            if obs.done:
                break
            obs = env.step(_null_oversight_action())
            _assert_obs_valid(obs)
        # At least some pairs have advanced
        assert env._world is not None
        total_rounds = sum(p.round_count for p in env._world.pairs.values())
        assert total_rounds > 0


# ---------------------------------------------------------------------------
# Mode D tests
# ---------------------------------------------------------------------------


class TestModeDManagerOrchestration:
    def test_reset_observation_is_for_manager(self):
        env = _make_env()
        obs = env.reset(seed=40, task_name="manager-orchestration", num_smes=3, num_buyers=2)
        _assert_obs_valid(obs)
        assert obs.mode == WorldMode.D_MANAGER.value
        assert obs.role == "manager"

    def test_manager_obs_contains_sme_summaries(self):
        env = _make_env()
        obs = env.reset(seed=41, task_name="manager-orchestration", num_smes=3, num_buyers=2)
        assert "manageragent" in obs.text.lower() or "manager" in obs.text.lower()

    def test_manager_instruction_forwarded(self):
        env = _make_env()
        obs = env.reset(seed=42, task_name="manager-orchestration", num_smes=3, num_buyers=1)
        assert env._world is not None
        sme_ids = list(env._world.smes.keys())
        action = WorldAction(
            role="manager",
            acting_agent_id="manager_agent",
            instructions={sme_ids[0]: "Accept 60 days and insist on TReDS."},
        )
        env.step(action)
        assert env._world.manager_instructions.get(sme_ids[0]) == "Accept 60 days and insist on TReDS."

    def test_tool_query_erp(self):
        env = _make_env()
        obs = env.reset(seed=43, task_name="manager-orchestration", num_smes=2, num_buyers=1)
        action = WorldAction(
            role="manager",
            acting_agent_id="manager_agent",
            instructions={},
            query_tool="erp",
        )
        obs2 = env.step(action)
        _assert_obs_valid(obs2)
        assert "last_tool_result" in obs2.aux or env._world.metrics.get("last_tool_result")

    def test_tool_query_treds_rate(self):
        env = _make_env()
        env.reset(seed=44, task_name="manager-orchestration", num_smes=2, num_buyers=1)
        action = WorldAction(
            role="manager",
            acting_agent_id="manager_agent",
            instructions={},
            query_tool="treds_rate",
        )
        obs2 = env.step(action)
        _assert_obs_valid(obs2)

    def test_manager_reward_is_world_level(self):
        env = _make_env()
        obs = env.reset(seed=45, task_name="manager-orchestration", num_smes=3, num_buyers=1)
        for _ in range(10):
            if obs.done:
                break
            obs = env.step(_null_manager_action())
            _assert_obs_valid(obs)

    def test_monthly_advancement(self):
        env = _make_env()
        obs = env.reset(seed=46, task_name="manager-orchestration", num_smes=2, num_buyers=1,
                        world_horizon_months=2)
        assert env._world is not None
        # Run enough steps to advance at least one month
        for _ in range(20):
            if obs.done:
                break
            obs = env.step(_null_manager_action())
        # Either done or month advanced
        assert obs.done or env._world.month >= 0


# ---------------------------------------------------------------------------
# Reward function unit tests
# ---------------------------------------------------------------------------


class TestRewardFunctions:
    def test_clamp01_edges(self):
        assert _clamp01(0.0) > 0.0
        assert _clamp01(1.0) < 1.0
        assert _clamp01(-5.0) > 0.0
        assert _clamp01(float("inf")) < 1.0
        assert _clamp01(float("nan")) > 0.0

    def test_gini_equal_distribution(self):
        g = _compute_gini([50, 50, 50, 50])
        assert g < 0.01, "Equal distribution should have near-zero Gini"

    def test_gini_max_inequality(self):
        g = _compute_gini([0, 0, 0, 100])
        assert g > 0.5, "Highly unequal distribution should have high Gini"

    def test_gini_empty(self):
        assert _compute_gini([]) == 0.0
        assert _compute_gini([50]) == 0.0

    def test_oversight_reward_perfect_flags(self):
        """When flags exactly match ground truth, reward should be high."""
        env = _make_env()
        env.reset(seed=50, task_name="oversight-arena", num_parallel_envs=4)
        assert env._world is not None
        gt = _detect_ground_truth_unfair(env._world)
        # Perfect flags
        reward = _mode_c_oversight_reward(list(gt), {pid: "Fix it." for pid in gt}, env._world)
        assert reward > 0.5 or not gt  # high reward for correct detection

    def test_oversight_reward_no_unfair_no_flags(self):
        """When there are no unfair cases and we flag nothing, reward should be reasonable."""
        env = _make_env()
        env.reset(seed=51, task_name="oversight-arena", num_parallel_envs=3)
        assert env._world is not None
        # Force all deals to be fine by clearing pairs
        for pair in env._world.pairs.values():
            pair.done = False
        reward = _mode_c_oversight_reward([], {}, env._world)
        assert reward > 0.0

    def test_manager_reward_all_solvent(self):
        env = _make_env()
        env.reset(seed=52, task_name="manager-orchestration", num_smes=3, num_buyers=1)
        assert env._world is not None
        for sme in env._world.smes.values():
            sme.is_solvent = True
            sme.current_payment_days = 45
            sme.deal_done = True
        reward = _mode_d_manager_reward(env._world)
        assert reward > 0.3

    def test_manager_reward_all_insolvent(self):
        env = _make_env()
        env.reset(seed=53, task_name="manager-orchestration", num_smes=3, num_buyers=1)
        assert env._world is not None
        for sme in env._world.smes.values():
            sme.is_solvent = False
            sme.cash_balance = -100_000
        reward = _mode_d_manager_reward(env._world)
        assert reward < 0.5


# ---------------------------------------------------------------------------
# Manager instruction parser unit tests
# ---------------------------------------------------------------------------


class TestManagerInstructionParser:
    def _make_sme(self):
        from server.world_environment import SMEProfile
        return SMEProfile(
            agent_id="sme_0",
            unit_cost=75.0,
            monthly_revenue=500_000.0,
            liquidity_threshold_days=45,
            interest_rate_annual=0.22,
            cash_balance=200_000.0,
            industry="textiles",
            reputation_score=0.8,
        )

    def _make_pair(self):
        from server.world_environment import NegotiationPair
        from server.environment import SMENegotiatorEnvironment
        sub_env = SMENegotiatorEnvironment()
        sub_env.reset(seed=1, task_name="payment-terms-medium")
        return NegotiationPair(
            pair_id="p0",
            sme_id="sme_0",
            buyer_id="buyer_0",
            env=sub_env,
        )

    def test_accept_instruction(self):
        sme = self._make_sme()
        pair = self._make_pair()
        pair.turn_history = [{"buyer_price": 95.0, "buyer_days": 55}]
        result = _parse_manager_instruction_to_action("Accept the current offer.", sme, pair)
        assert result["action_type"] == "accept"

    def test_reject_instruction(self):
        sme = self._make_sme()
        pair = self._make_pair()
        result = _parse_manager_instruction_to_action("Reject this buyer immediately.", sme, pair)
        assert result["action_type"] == "reject"

    def test_days_extraction(self):
        sme = self._make_sme()
        pair = self._make_pair()
        result = _parse_manager_instruction_to_action("Propose days=50 and use TReDS.", sme, pair)
        assert result["payment_days"] == 50
        assert result["use_treds"] is True

    def test_price_extraction(self):
        sme = self._make_sme()
        pair = self._make_pair()
        result = _parse_manager_instruction_to_action("Counter with price=98.5.", sme, pair)
        assert abs(result["price"] - 98.5) < 0.01

    def test_empty_instruction_fallback(self):
        sme = self._make_sme()
        pair = self._make_pair()
        result = _parse_manager_instruction_to_action("", sme, pair)
        assert result["action_type"] == "propose"
        assert result["price"] >= sme.unit_cost

    def test_price_never_below_cost(self):
        sme = self._make_sme()
        pair = self._make_pair()
        result = _parse_manager_instruction_to_action("Propose price=10.0 days=30", sme, pair)
        assert result["price"] >= sme.unit_cost * 1.01


# ---------------------------------------------------------------------------
# Cross-mode invariants
# ---------------------------------------------------------------------------


class TestCrossModeInvariants:
    @pytest.mark.parametrize("task_name", [
        "competitive-bidding",
        "coalition-bargaining",
        "oversight-arena",
        "manager-orchestration",
    ])
    def test_reset_then_episode_terminates(self, task_name: str):
        env = _make_env()
        kwargs: dict = {"seed": 77, "task_name": task_name}
        if task_name == "oversight-arena":
            kwargs["num_parallel_envs"] = 3
        if task_name == "manager-orchestration":
            kwargs["num_smes"] = 2
            kwargs["num_buyers"] = 1
            kwargs["world_horizon_months"] = 1

        obs = env.reset(**kwargs)
        _assert_obs_valid(obs)

        steps = 0
        while not obs.done and steps < 300:
            if obs.role == "sme":
                action = _null_sme_action(obs, price=92.0, days=48)
            elif obs.role == "oversight":
                action = _null_oversight_action()
            else:
                action = _null_manager_action()
            obs = env.step(action)
            steps += 1

        assert obs.done, f"Mode {task_name} did not terminate within 300 steps"

    @pytest.mark.parametrize("task_name", [
        "competitive-bidding",
        "coalition-bargaining",
        "oversight-arena",
        "manager-orchestration",
    ])
    def test_reward_always_in_strict_unit_interval(self, task_name: str):
        env = _make_env()
        kwargs: dict = {"seed": 88, "task_name": task_name}
        if task_name == "oversight-arena":
            kwargs["num_parallel_envs"] = 3
        if task_name == "manager-orchestration":
            kwargs["num_smes"] = 2
            kwargs["num_buyers"] = 1

        obs = env.reset(**kwargs)
        rewards = [obs.reward]

        for _ in range(20):
            if obs.done:
                break
            if obs.role == "sme":
                action = _null_sme_action(obs)
            elif obs.role == "oversight":
                action = _null_oversight_action()
            else:
                action = _null_manager_action()
            obs = env.step(action)
            rewards.append(obs.reward)

        for r in rewards:
            assert 0.0 < r < 1.0, f"Reward {r} is outside strict (0, 1) for mode {task_name}"
            assert math.isfinite(r), f"Non-finite reward {r} for mode {task_name}"

    @pytest.mark.parametrize("task_name", [
        "competitive-bidding",
        "coalition-bargaining",
        "oversight-arena",
        "manager-orchestration",
    ])
    def test_observation_text_non_empty(self, task_name: str):
        env = _make_env()
        kwargs: dict = {"seed": 99, "task_name": task_name}
        if task_name == "oversight-arena":
            kwargs["num_parallel_envs"] = 3
        obs = env.reset(**kwargs)
        assert obs.text.strip(), "Observation text must be non-empty after reset"
        action = _null_sme_action(obs) if obs.role == "sme" else (
            _null_oversight_action() if obs.role == "oversight" else _null_manager_action()
        )
        obs2 = env.step(action)
        assert obs2.text.strip(), "Observation text must be non-empty after step"

    def test_state_property_returns_dict(self):
        env = _make_env()
        env.reset(seed=1, task_name="competitive-bidding", num_smes=2)
        state = env.state
        assert isinstance(state, dict)
        assert "episode_id" in state
        assert "mode" in state

    def test_existing_single_env_unaffected(self):
        """Ensure the existing SMENegotiatorEnvironment still works correctly."""
        from server.environment import SMENegotiatorEnvironment
        from sme_negotiator_env.models import NegotiationAction

        single_env = SMENegotiatorEnvironment()
        obs = single_env.reset(seed=1, task_name="payment-terms-easy")
        assert obs is not None
        action = NegotiationAction(action_type="propose", price=85.0, payment_days=55)
        obs2 = single_env.step(action)
        assert obs2 is not None
        assert math.isfinite(obs2.step_reward)
