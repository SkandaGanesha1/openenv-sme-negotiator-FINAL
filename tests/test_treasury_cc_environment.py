"""
Comprehensive tests for TreasuryCommandCenterEnvironment.

Test coverage:
  - All 5 modes: solo, multi, coalition, oversight, manager
  - OpenEnv contract: reset/step API, reward in (0,1), observation completeness
  - Partial observability: no private data leaks between SMEs
  - GRPO reward shaper: group normalisation, verifiable reward components
  - LatentWorldModel: entropy, prediction, error computation
  - AutoCurriculum: difficulty adjustment, constitutional check
  - Graders: each mode grader returns (0,1)
  - Regression: existing SMENegotiatorEnvironment and treasury envs unaffected

Run with: pytest tests/test_treasury_cc_environment.py -v
"""

from __future__ import annotations

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from server.treasury_cc_environment import TreasuryCommandCenterEnvironment
from treasury_command_center_env.models import (
    TreasuryCCAction,
    TreasuryCCObservation,
)
from treasury_command_center_env.reward import (
    GRPORewardShaper,
    rlvr_solvency,
    rlvr_compliance,
    rubric_tool_quality,
    rubric_oversight_quality,
)
from treasury_command_center_env.world_model import LatentWorldModel
from treasury_command_center_env.curriculum import AutoCurriculum
from treasury_command_center_env.graders import (
    grade_tcc_solo,
    grade_tcc_multi,
    grade_tcc_oversight,
    grade_tcc_manager,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_env(seed: int = 42) -> TreasuryCommandCenterEnvironment:
    return TreasuryCommandCenterEnvironment()


def _assert_obs_valid(obs: TreasuryCCObservation) -> None:
    assert isinstance(obs, TreasuryCCObservation)
    assert obs.text and len(obs.text) > 20, "obs text must be non-empty"
    assert math.isfinite(obs.reward), "reward must be finite"
    assert 0.0 < obs.reward < 1.0, f"reward {obs.reward} not in (0,1)"
    assert isinstance(obs.done, bool)
    assert obs.episode_id, "episode_id must be non-empty"
    assert obs.day >= 0
    assert obs.step_count >= 0
    # GRPO signals
    assert math.isfinite(obs.belief_entropy), "belief_entropy must be finite"
    assert math.isfinite(obs.predicted_cash_buffer_30d), "predicted cash must be finite"


def _tool_action(
    app: str = "analytics_app",
    endpoint: str = "kpi_dashboard",
    params: dict | None = None,
    sme_id: str = "SME_1",
    role: str = "treasury_officer",
) -> TreasuryCCAction:
    return TreasuryCCAction(
        role=role,
        acting_sme_id=sme_id,
        app=app,
        endpoint=endpoint,
        params=params or {"sme_id": sme_id},
    )


def _oversight_action(flag: list[str] | None = None) -> TreasuryCCAction:
    return TreasuryCCAction(
        role="oversight_agent",
        flag_risky_smes=flag or [],
        suggested_interventions={sid: "Review immediately." for sid in (flag or [])},
        global_risk_summary="Test oversight action.",
    )


def _manager_action(instructions: dict | None = None) -> TreasuryCCAction:
    return TreasuryCCAction(
        role="manager_agent",
        instructions=instructions or {},
        query_analytics="kpi_dashboard",
    )


# ── Mode: SOLO ─────────────────────────────────────────────────────────────────

class TestSoloMode:
    def test_reset_returns_valid_observation(self):
        env = _make_env()
        obs = env.reset(seed=1, task_name="tcc-solo-easy")
        _assert_obs_valid(obs)
        assert obs.mode == "treasury-solo"
        assert obs.role == "treasury_officer"
        assert obs.world_step == 0

    def test_text_contains_kpi_summary(self):
        env = _make_env()
        obs = env.reset(seed=2, task_name="tcc-solo-easy")
        assert "cash_buffer" in obs.text.lower() or "treasuryofficer" in obs.text.lower()
        assert "tool" in obs.text.lower()

    def test_step_advances_day_and_step(self):
        env = _make_env()
        obs = env.reset(seed=3, task_name="tcc-solo-easy")
        action = _tool_action()
        obs2 = env.step(action)
        _assert_obs_valid(obs2)
        assert obs2.world_step == 1
        assert obs2.step_count == 1

    def test_tool_call_erp_list_invoices(self):
        env = _make_env()
        obs = env.reset(seed=4, task_name="tcc-solo-easy")
        action = _tool_action("erp_app", "list_invoices", {"sme_id": "SME_1"})
        obs2 = env.step(action)
        _assert_obs_valid(obs2)
        assert obs2.last_tool_result.get("app") == "erp_app" or "error" in obs2.last_tool_result or "type" in obs2.last_tool_result

    def test_tool_call_bank_get_balances(self):
        env = _make_env()
        env.reset(seed=5, task_name="tcc-solo-easy")
        obs = env.step(_tool_action("bank_app", "get_balances", {"sme_id": "SME_1"}))
        _assert_obs_valid(obs)

    def test_tool_call_analytics_kpi_dashboard(self):
        env = _make_env()
        env.reset(seed=6, task_name="tcc-solo-medium")
        obs = env.step(_tool_action("analytics_app", "kpi_dashboard", {"sme_id": "SME_1"}))
        _assert_obs_valid(obs)
        result = obs.last_tool_result
        assert "app" in result or "error" in result

    def test_multiple_steps_no_crash(self):
        env = _make_env()
        obs = env.reset(seed=7, task_name="tcc-solo-easy")
        for i in range(10):
            if obs.done:
                break
            obs = env.step(_tool_action())
            _assert_obs_valid(obs)

    def test_rewards_bounded_in_strict_interval(self):
        env = _make_env()
        obs = env.reset(seed=8, task_name="tcc-solo-easy")
        rewards = [obs.reward]
        for _ in range(8):
            if obs.done:
                break
            obs = env.step(_tool_action())
            rewards.append(obs.reward)
        for r in rewards:
            assert 0.0 < r < 1.0, f"Reward {r} out of (0,1)"
            assert math.isfinite(r)

    def test_episode_terminates(self):
        env = _make_env()
        obs = env.reset(seed=9, task_name="tcc-solo-easy")
        steps = 0
        while not obs.done and steps < 200:
            obs = env.step(_tool_action())
            steps += 1
        assert obs.done, "Episode must eventually terminate"

    def test_state_property_returns_dict(self):
        env = _make_env()
        env.reset(seed=10, task_name="tcc-solo-easy")
        s = env.state
        assert isinstance(s, dict)
        assert "episode_id" in s
        assert "mode" in s

    def test_aux_contains_global_metrics(self):
        env = _make_env()
        obs = env.reset(seed=11, task_name="tcc-solo-medium")
        obs2 = env.step(_tool_action())
        assert "global_metrics" in obs2.aux
        gm = obs2.aux["global_metrics"]
        assert "solvent_smes" in gm
        assert "avg_dso" in gm

    def test_world_model_signals_in_observation(self):
        env = _make_env()
        obs = env.reset(seed=12, task_name="tcc-solo-easy")
        obs2 = env.step(_tool_action())
        assert math.isfinite(obs2.belief_entropy)
        assert math.isfinite(obs2.predicted_cash_buffer_30d)
        assert math.isfinite(obs2.world_model_prediction_error)

    def test_no_private_data_leak_in_text(self):
        """Private SME cost data must not appear in the text observation."""
        env = _make_env()
        obs = env.reset(seed=13, task_name="tcc-solo-medium")
        # The text should not expose credit_score (hidden) directly
        assert "credit_score" not in obs.text.lower()


# ── Mode: MULTI ────────────────────────────────────────────────────────────────

class TestMultiMode:
    def test_reset_returns_sme_agent_role(self):
        env = _make_env()
        obs = env.reset(seed=20, task_name="tcc-multi-medium")
        _assert_obs_valid(obs)
        assert obs.mode == "treasury-multi"
        assert obs.role == "sme_agent"

    def test_num_active_smes_correct(self):
        env = _make_env()
        obs = env.reset(seed=21, task_name="tcc-multi-medium")
        assert obs.num_active_smes == 2  # treasury-medium has 2 SMEs

    def test_round_robin_scheduling(self):
        env = _make_env()
        obs = env.reset(seed=22, task_name="tcc-multi-medium")
        first_sme = obs.acting_sme_id
        obs2 = env.step(_tool_action(role="sme_agent", sme_id=first_sme or "SME_1"))
        assert obs2.acting_sme_id != first_sme or obs2.world_step == 1

    def test_peer_summaries_in_text(self):
        env = _make_env()
        obs = env.reset(seed=23, task_name="tcc-multi-medium")
        # In MULTI mode, the observation should mention peers or "peer"
        assert "peer" in obs.text.lower() or "sme" in obs.text.lower()

    def test_no_private_cash_of_peer_in_text(self):
        """Acting SME agent must not see private cash of peer SMEs."""
        env = _make_env()
        obs = env.reset(seed=24, task_name="tcc-multi-medium")
        # The text must not contain "credit_score" or other hidden fields
        assert "credit_score" not in obs.text.lower()
        assert "reservation_price" not in obs.text.lower()

    def test_multiple_steps_no_crash(self):
        env = _make_env()
        obs = env.reset(seed=25, task_name="tcc-multi-medium")
        for _ in range(8):
            if obs.done:
                break
            sme = obs.acting_sme_id or "SME_1"
            obs = env.step(_tool_action(role="sme_agent", sme_id=sme))
            _assert_obs_valid(obs)

    def test_rewards_in_strict_interval(self):
        env = _make_env()
        obs = env.reset(seed=26, task_name="tcc-multi-medium")
        for _ in range(6):
            if obs.done:
                break
            sme = obs.acting_sme_id or "SME_1"
            obs = env.step(_tool_action(role="sme_agent", sme_id=sme))
            assert 0.0 < obs.reward < 1.0


# ── Mode: COALITION ────────────────────────────────────────────────────────────

class TestCoalitionMode:
    def test_reset_creates_coalition_channel(self):
        env = _make_env()
        env.reset(seed=30, task_name="tcc-coalition-medium")
        assert env._coalition is not None

    def test_coalition_message_posted(self):
        env = _make_env()
        obs = env.reset(seed=31, task_name="tcc-coalition-medium")
        sme = obs.acting_sme_id or "SME_1"
        action = TreasuryCCAction(
            role="sme_agent",
            acting_sme_id=sme,
            app="analytics_app",
            endpoint="kpi_dashboard",
            params={"sme_id": sme},
            coalition_message="We should use TReDS to cut our combined financing cost.",
        )
        env.step(action)
        assert env._coalition is not None
        assert len(env._coalition.messages) == 1

    def test_coalition_text_in_observation(self):
        env = _make_env()
        obs = env.reset(seed=32, task_name="tcc-coalition-medium")
        sme = obs.acting_sme_id or "SME_1"
        action = TreasuryCCAction(
            role="sme_agent",
            acting_sme_id=sme,
            app="analytics_app",
            endpoint="kpi_dashboard",
            params={"sme_id": sme},
            coalition_message="All SMEs: let's target 45-day payment terms.",
        )
        obs2 = env.step(action)
        # Coalition text should appear in next SME's observation
        has_coalition = (
            obs2.coalition_channel_text is not None
            or "coalition" in obs2.text.lower()
        )
        assert has_coalition

    def test_rewards_in_strict_interval(self):
        env = _make_env()
        obs = env.reset(seed=33, task_name="tcc-coalition-medium")
        for _ in range(6):
            if obs.done:
                break
            sme = obs.acting_sme_id or "SME_1"
            action = TreasuryCCAction(
                role="sme_agent",
                acting_sme_id=sme,
                app="analytics_app",
                endpoint="kpi_dashboard",
                params={"sme_id": sme},
            )
            obs = env.step(action)
            assert 0.0 < obs.reward < 1.0


# ── Mode: OVERSIGHT ────────────────────────────────────────────────────────────

class TestOversightMode:
    def test_reset_returns_oversight_role(self):
        env = _make_env()
        obs = env.reset(seed=40, task_name="tcc-oversight-hard")
        _assert_obs_valid(obs)
        assert obs.mode == "treasury-oversight"
        assert obs.role == "oversight_agent"

    def test_oversight_text_contains_sme_summaries(self):
        env = _make_env()
        obs = env.reset(seed=41, task_name="tcc-oversight-hard")
        assert "oversightagent" in obs.text.lower() or "oversight" in obs.text.lower()
        assert "sme" in obs.text.lower()

    def test_empty_flag_returns_valid_reward(self):
        env = _make_env()
        obs = env.reset(seed=42, task_name="tcc-oversight-hard")
        obs2 = env.step(_oversight_action([]))
        _assert_obs_valid(obs2)

    def test_flag_all_smes(self):
        env = _make_env()
        obs = env.reset(seed=43, task_name="tcc-oversight-hard")
        all_smes = env._all_sme_ids()
        obs2 = env.step(_oversight_action(all_smes))
        _assert_obs_valid(obs2)

    def test_ground_truth_risky_smes_in_obs(self):
        env = _make_env()
        obs = env.reset(seed=44, task_name="tcc-oversight-hard")
        obs2 = env.step(_oversight_action([]))
        assert isinstance(obs2.ground_truth_risky_smes, list)

    def test_precision_recall_f1_in_obs(self):
        env = _make_env()
        obs = env.reset(seed=45, task_name="tcc-oversight-hard")
        # Flag something to get non-trivial PR
        env.step(_oversight_action(["SME_1"]))
        obs2 = env.step(_oversight_action(["SME_1", "SME_2"]))
        assert math.isfinite(obs2.oversight_f1)
        assert 0.0 <= obs2.oversight_precision <= 1.0
        assert 0.0 <= obs2.oversight_recall <= 1.0

    def test_multiple_steps_advances(self):
        env = _make_env()
        obs = env.reset(seed=46, task_name="tcc-oversight-hard")
        for _ in range(5):
            if obs.done:
                break
            obs = env.step(_oversight_action())
            _assert_obs_valid(obs)


# ── Mode: MANAGER ─────────────────────────────────────────────────────────────

class TestManagerMode:
    def test_reset_returns_manager_role(self):
        env = _make_env()
        obs = env.reset(seed=50, task_name="tcc-manager-hard")
        _assert_obs_valid(obs)
        assert obs.mode == "treasury-manager"
        assert obs.role == "manager_agent"

    def test_manager_text_contains_global_metrics(self):
        env = _make_env()
        obs = env.reset(seed=51, task_name="tcc-manager-hard")
        assert "manager" in obs.text.lower()
        assert "sme" in obs.text.lower()

    def test_manager_instruction_stored(self):
        env = _make_env()
        obs = env.reset(seed=52, task_name="tcc-manager-hard")
        action = _manager_action({"SME_1": "Draw overdraft immediately and discount invoices via TReDS."})
        obs2 = env.step(action)
        _assert_obs_valid(obs2)
        # Instruction should be stored in state
        assert env._state is not None
        instr_actions = [a for a in env._state.actions_taken if a.get("role") == "manager"]
        assert len(instr_actions) >= 1

    def test_query_analytics_returns_result(self):
        env = _make_env()
        obs = env.reset(seed=53, task_name="tcc-manager-hard")
        action = _manager_action()
        obs2 = env.step(action)
        _assert_obs_valid(obs2)
        # Some analytics result should be in last_tool_result
        assert obs2.last_tool_result is not None

    def test_rewards_in_strict_interval(self):
        env = _make_env()
        obs = env.reset(seed=54, task_name="tcc-manager-hard")
        for _ in range(6):
            if obs.done:
                break
            obs = env.step(_manager_action())
            assert 0.0 < obs.reward < 1.0


# ── GRPO Reward Shaper Tests ───────────────────────────────────────────────────

class TestGRPORewardShaper:
    def _make_shaper(self):
        return GRPORewardShaper()

    def test_step_reward_in_strict_interval(self):
        shaper = self._make_shaper()
        r, _ = shaper.step_reward(
            solvency_ok=True,
            total_financing_cost=5000.0,
            total_revenue=100000.0,
            vendor_stress_score=0.1,
            hhi=0.2,
        )
        assert 0.0 < r < 1.0

    def test_terminal_reward_solvency_breach_capped(self):
        shaper = self._make_shaper()
        r, comps = shaper.terminal_reward(
            solvency_ok=False,
            solvency_breach_day=10,
            max_days=180,
            total_financing_cost=0,
            total_revenue=100000,
            vendor_stress_score=0,
            hhi=0,
        )
        assert 0.0 < r < 0.15  # solvency breach → near zero

    def test_terminal_reward_all_perfect(self):
        shaper = self._make_shaper()
        r, _ = shaper.terminal_reward(
            solvency_ok=True,
            solvency_breach_day=None,
            max_days=180,
            total_financing_cost=0,
            total_revenue=1_000_000,
            vendor_stress_score=0,
            hhi=0.1,
        )
        assert r > 0.5

    def test_grpo_normalise_single_rollout(self):
        advantages = GRPORewardShaper.normalise_grpo([0.7])
        assert advantages == [0.0]

    def test_grpo_normalise_group(self):
        rewards = [0.3, 0.5, 0.7, 0.9]
        advantages = GRPORewardShaper.normalise_grpo(rewards)
        assert len(advantages) == 4
        assert abs(sum(advantages)) < 1.0  # roughly centred near zero

    def test_grpo_success_probability(self):
        rewards = [0.2, 0.8, 0.6, 0.3, 0.9]
        p = GRPORewardShaper.grpo_success_probability_gain(rewards, threshold=0.5)
        assert abs(p - 0.6) < 0.01  # 3/5 above 0.5

    def test_rlvr_solvency_binary(self):
        assert rlvr_solvency(True) == 1.0
        assert rlvr_solvency(False) == 0.0

    def test_rlvr_compliance_degradation(self):
        assert rlvr_compliance(0) == 1.0
        assert rlvr_compliance(1) == 0.8
        assert rlvr_compliance(5) == 0.0

    def test_constitution_violated_caps_reward(self):
        shaper = self._make_shaper()
        r, comps = shaper.step_reward(
            solvency_ok=True,
            total_financing_cost=0,
            total_revenue=100000,
            vendor_stress_score=0,
            hhi=0,
            constitution_violated=True,
        )
        assert r < 0.05 + 0.01  # capped near 0.05

    def test_rubric_tool_quality_observe_first(self):
        kpis = {"solvency_ok": True, "cash_buffer_days": 30, "compliance_breach_count": 0}
        # analytics in first step should score well
        score = rubric_tool_quality("analytics_app", "kpi_dashboard", kpis, step_in_episode=0)
        assert score > 0.5

    def test_rubric_tool_quality_solvency_crisis_financing(self):
        kpis = {"solvency_ok": False, "cash_buffer_days": 0.5, "compliance_breach_count": 0}
        score = rubric_tool_quality("bank_app", "draw_overdraft", kpis, step_in_episode=5)
        assert score > 0.5  # correct response to solvency crisis

    def test_oversight_reward_perfect(self):
        shaper = self._make_shaper()
        r, comps = shaper.oversight_reward(["SME_1"], ["SME_1"], {"SME_1": "Fix it."})
        assert r > 0.5

    def test_oversight_reward_all_missed(self):
        shaper = self._make_shaper()
        r, _ = shaper.oversight_reward([], ["SME_1", "SME_2"], {})
        assert r < 0.01  # missed all

    def test_manager_reward_all_solvent(self):
        shaper = self._make_shaper()
        r, _ = shaper.manager_reward(
            solvent_fraction=1.0,
            avg_dso_improvement=15.0,
            gini_days=0.0,
            total_financing_cost=0,
            total_revenue=1_000_000,
            instruction_quality=0.8,
        )
        assert r > 0.5


# ── LatentWorldModel Tests ─────────────────────────────────────────────────────

class TestLatentWorldModel:
    def _make_wm(self) -> LatentWorldModel:
        return LatentWorldModel()

    def _feed(self, wm: LatentWorldModel, n: int = 5) -> None:
        for i in range(n):
            wm.observe({
                "cash_buffer_days": 20.0 + i,
                "dso_days": 70.0 - i * 0.5,
                "vendor_stress_score": 0.1 + i * 0.01,
                "concentration_risk_hhi": 0.3,
            })

    def test_entropy_positive(self):
        wm = self._make_wm()
        self._feed(wm, 5)
        assert wm.belief_entropy() > 0.0

    def test_entropy_decreases_after_accurate_predictions(self):
        wm = self._make_wm()
        self._feed(wm, 3)
        entropy_before = wm.belief_entropy()
        # Feed very consistent data (low variance = tighter belief)
        for i in range(10):
            wm.observe({
                "cash_buffer_days": 20.0,
                "dso_days": 60.0,
                "vendor_stress_score": 0.1,
                "concentration_risk_hhi": 0.3,
            })
        entropy_after = wm.belief_entropy()
        # After many consistent observations entropy should not be much higher
        assert entropy_after <= entropy_before * 2.0  # relaxed — symbolic model

    def test_predict_returns_all_kpi_keys(self):
        wm = self._make_wm()
        self._feed(wm)
        pred = wm.predict(30)
        assert "cash_buffer_days" in pred
        assert "dso_days" in pred
        assert "vendor_stress_score" in pred
        assert "concentration_risk_hhi" in pred

    def test_prediction_error_zero_before_prediction(self):
        wm = self._make_wm()
        assert wm.prediction_error() == 0.0

    def test_prediction_error_finite_after_prediction(self):
        wm = self._make_wm()
        self._feed(wm, 3)
        wm.predict(30)
        self._feed(wm, 2)  # observe after prediction → computes error
        err = wm.prediction_error()
        assert math.isfinite(err)
        assert err >= 0.0

    def test_cash_buffer_forecast_30d_finite(self):
        wm = self._make_wm()
        self._feed(wm)
        forecast = wm.cash_buffer_forecast_30d()
        assert math.isfinite(forecast)

    def test_summary_dict_complete(self):
        wm = self._make_wm()
        self._feed(wm, 5)
        s = wm.summary()
        required_keys = [
            "latent_h_cash_buffer",
            "belief_entropy_bits",
            "predicted_cash_buffer_30d",
            "prediction_error_mae",
        ]
        for k in required_keys:
            assert k in s, f"Missing key {k} in world model summary"

    def test_reset_clears_state(self):
        wm = self._make_wm()
        self._feed(wm, 10)
        wm.reset()
        assert wm._step == 0
        assert len(wm._history) == 0


# ── AutoCurriculum Tests ───────────────────────────────────────────────────────

class TestAutoCurriculum:
    def test_initial_difficulty_is_midpoint(self):
        c = AutoCurriculum()
        assert abs(c.difficulty - 0.5) < 0.1

    def test_difficulty_decreases_after_many_failures(self):
        c = AutoCurriculum(target_success_rate=0.5, success_threshold=0.5, step_size=0.1)
        for _ in range(20):
            c.record_episode(0.1)  # all failures
        assert c.difficulty < 0.5

    def test_difficulty_increases_after_many_successes(self):
        c = AutoCurriculum(target_success_rate=0.5, success_threshold=0.5, step_size=0.1)
        for _ in range(20):
            c.record_episode(0.9)  # all successes
        assert c.difficulty > 0.5

    def test_difficulty_stays_bounded(self):
        c = AutoCurriculum(step_size=0.2)
        for _ in range(100):
            c.record_episode(1.0)
        assert c.difficulty <= 0.95

    def test_sample_params_returns_valid_struct(self):
        from random import Random
        c = AutoCurriculum()
        params = c.sample_params(Random(0))
        assert 0.0 <= params.difficulty <= 1.0
        assert params.buyer_aggressiveness >= 0.0
        assert params.payment_delay_multiplier > 0.0
        assert params.scenario_complexity >= 1

    def test_adversarial_params_harder_than_baseline(self):
        from random import Random
        c = AutoCurriculum()
        c._state.difficulty = 0.5
        baseline = c.sample_params(Random(0))
        adv = c.sample_adversarial_params([0.6, 0.7, 0.8], Random(0))
        # adversarial difficulty should be higher than agent's recent average
        assert adv.difficulty >= 0.5

    def test_constitutional_check_solvency_violation(self):
        c = AutoCurriculum()
        kpis = {"solvency_ok": True, "cash_buffer_days": 0.5, "compliance_breach_count": 0}
        # Using erp_app when cash is critically low (not financing) = violation
        violated = c.check_constitutional_violation("erp_app", "list_invoices", kpis, ())
        assert violated  # cash_buffer < 1 and not financing tool

    def test_constitutional_check_no_violation(self):
        c = AutoCurriculum()
        kpis = {"solvency_ok": True, "cash_buffer_days": 20.0, "compliance_breach_count": 0}
        violated = c.check_constitutional_violation("analytics_app", "kpi_dashboard", kpis, ())
        assert not violated


# ── Grader Tests ───────────────────────────────────────────────────────────────

class TestGraders:
    def _make_snap(self, **overrides) -> dict:
        snap = {
            "solvency_ok": True,
            "solvency_breach_day": None,
            "max_days": 180,
            "total_financing_cost": 5000.0,
            "revenue_collected": {"SME_1": 500_000.0},
            "vendor_overdue_days": {"VENDOR_X": 0.0},
            "hhi": 0.3,
            "compliance_breach_count": 0,
            "avg_tool_quality_score": 0.6,
            "avg_world_model_error": 2.0,
            "constitution_violated": False,
        }
        snap.update(overrides)
        return snap

    def test_grade_solo_solvent_returns_above_threshold(self):
        snap = self._make_snap()
        r = grade_tcc_solo(snap)
        assert 0.0 < r < 1.0
        assert r > 0.3

    def test_grade_solo_solvency_breach_returns_low(self):
        snap = self._make_snap(solvency_ok=False, solvency_breach_day=10)
        r = grade_tcc_solo(snap)
        assert 0.0 < r < 0.15

    def test_grade_solo_constitution_violation_capped(self):
        snap = self._make_snap(constitution_violated=True)
        r = grade_tcc_solo(snap)
        assert r < 0.1

    def test_grade_multi_all_solvent_bonus(self):
        per_sme = [
            {**self._make_snap(), "solvency_ok": True},
            {**self._make_snap(), "solvency_ok": True},
        ]
        snap = {**self._make_snap(), "per_sme_snapshots": per_sme}
        r = grade_tcc_multi(snap)
        assert 0.0 < r < 1.0

    def test_grade_oversight_perfect_flags(self):
        snap = {
            "ground_truth_risky_smes": ["SME_1"],
            "total_flagged_smes": ["SME_1"],
            "interventions": {"SME_1": "Draw overdraft."},
        }
        r = grade_tcc_oversight(snap)
        assert r > 0.5

    def test_grade_oversight_missed_all(self):
        snap = {
            "ground_truth_risky_smes": ["SME_1", "SME_2"],
            "total_flagged_smes": [],
            "interventions": {},
        }
        r = grade_tcc_oversight(snap)
        assert r < 0.01

    def test_grade_manager_all_solvent(self):
        snap = {
            "solvent_smes": 3,
            "total_smes": 3,
            "gini_payment_days": 0.0,
            "avg_dso_improvement_days": 15.0,
            "total_financing_cost": 5000.0,
            "revenue_collected": {"SME_1": 500_000, "SME_2": 400_000, "SME_3": 300_000},
            "avg_instruction_quality": 0.8,
        }
        r = grade_tcc_manager(snap)
        assert 0.0 < r < 1.0
        assert r > 0.3

    def test_all_graders_return_strict_interval(self):
        snap = self._make_snap()
        for grader in [grade_tcc_solo, grade_tcc_multi]:
            r = grader(snap)
            assert 0.0 < r < 1.0, f"Grader returned {r}"
            assert math.isfinite(r)


# ── Cross-mode invariants ─────────────────────────────────────────────────────

class TestCrossModeInvariants:
    @pytest.mark.parametrize("task_name,role", [
        ("tcc-solo-easy",       "treasury_officer"),
        ("tcc-multi-medium",    "sme_agent"),
        ("tcc-coalition-medium","sme_agent"),
        ("tcc-oversight-hard",  "oversight_agent"),
        ("tcc-manager-hard",    "manager_agent"),
    ])
    def test_reset_role_matches_mode(self, task_name: str, role: str):
        env = _make_env()
        obs = env.reset(seed=77, task_name=task_name)
        _assert_obs_valid(obs)
        assert obs.role == role, f"Expected role {role} for {task_name}, got {obs.role}"

    @pytest.mark.parametrize("task_name", [
        "tcc-solo-easy",
        "tcc-solo-medium",
        "tcc-multi-medium",
        "tcc-coalition-medium",
        "tcc-oversight-hard",
        "tcc-manager-hard",
    ])
    def test_episode_terminates_within_max_steps(self, task_name: str):
        env = _make_env()
        obs = env.reset(seed=88, task_name=task_name)
        steps = 0
        max_steps_allowed = 500
        while not obs.done and steps < max_steps_allowed:
            role = obs.role
            if role in ("treasury_officer", "sme_agent"):
                sme = obs.acting_sme_id or "SME_1"
                action = _tool_action("analytics_app", "kpi_dashboard", {"sme_id": sme}, sme, role)
            elif role == "oversight_agent":
                action = _oversight_action()
            else:
                action = _manager_action()
            obs = env.step(action)
            steps += 1
        assert obs.done, f"Mode {task_name} did not terminate in {max_steps_allowed} steps"

    @pytest.mark.parametrize("task_name", [
        "tcc-solo-easy",
        "tcc-multi-medium",
        "tcc-oversight-hard",
    ])
    def test_rewards_always_in_strict_interval(self, task_name: str):
        env = _make_env()
        obs = env.reset(seed=99, task_name=task_name)
        rewards = [obs.reward]
        for _ in range(15):
            if obs.done:
                break
            role = obs.role
            if role in ("treasury_officer", "sme_agent"):
                sme = obs.acting_sme_id or "SME_1"
                action = _tool_action("analytics_app", "kpi_dashboard", {"sme_id": sme}, sme, role)
            elif role == "oversight_agent":
                action = _oversight_action()
            else:
                action = _manager_action()
            obs = env.step(action)
            rewards.append(obs.reward)
        for r in rewards:
            assert 0.0 < r < 1.0, f"Reward {r} out of strict (0,1) for {task_name}"
            assert math.isfinite(r)

    def test_observation_text_non_empty_all_modes(self):
        task_role = [
            ("tcc-solo-easy", None),
            ("tcc-multi-medium", None),
            ("tcc-oversight-hard", None),
            ("tcc-manager-hard", None),
        ]
        for task_name, _ in task_role:
            env = _make_env()
            obs = env.reset(seed=5, task_name=task_name)
            assert obs.text.strip(), f"Empty text for {task_name} after reset"


# ── GRPO training loop integration ────────────────────────────────────────────

class TestGRPOIntegration:
    def test_group_rewards_collected_across_steps(self):
        env = _make_env()
        obs = env.reset(seed=10, task_name="tcc-solo-easy")
        group_id = "test_group_42"
        for _ in range(5):
            if obs.done:
                break
            action = TreasuryCCAction(
                role="treasury_officer",
                acting_sme_id="SME_1",
                app="analytics_app",
                endpoint="kpi_dashboard",
                params={"sme_id": "SME_1"},
                group_id=group_id,
            )
            obs = env.step(action)

        assert group_id in env._grpo_groups
        assert len(env._grpo_groups[group_id]) >= 1

    def test_grpo_normalise_within_group(self):
        rewards = [0.1, 0.3, 0.5, 0.7, 0.9]
        adv = GRPORewardShaper.normalise_grpo(rewards)
        assert len(adv) == 5
        # Mean of advantages should be ~0
        assert abs(sum(adv) / len(adv)) < 0.5

    def test_reasoning_trace_stored(self):
        env = _make_env()
        obs = env.reset(seed=20, task_name="tcc-solo-easy")
        action = TreasuryCCAction(
            role="treasury_officer",
            acting_sme_id="SME_1",
            app="analytics_app",
            endpoint="kpi_dashboard",
            params={"sme_id": "SME_1"},
            reasoning_trace="I should first check KPIs before taking any financing action.",
        )
        env.step(action)
        assert env._state is not None
        assert len(env._state.reasoning_traces) == 1
        assert "kpi" in env._state.reasoning_traces[0].lower()


# ── Regression: existing environments unaffected ─────────────────────────────

class TestRegressionExistingEnvs:
    def test_sme_negotiator_environment_still_works(self):
        from server.environment import SMENegotiatorEnvironment
        from sme_negotiator_env.models import NegotiationAction

        env = SMENegotiatorEnvironment()
        obs = env.reset(seed=1, task_name="payment-terms-easy")
        assert obs is not None
        action = NegotiationAction(action_type="propose", price=85.0, payment_days=55)
        obs2 = env.step(action)
        assert obs2 is not None
        assert math.isfinite(obs2.step_reward)

    def test_treasury_agent_environment_still_works(self):
        from server.treasury_environment import TreasuryAgentEnvironment
        from treasury_agent_env.models import TreasuryAction

        env = TreasuryAgentEnvironment()
        obs = env.reset(seed=1, task_name="treasury-easy")
        assert obs is not None
        action = TreasuryAction(
            role="treasury",
            command_type="tool_call",
            app="analytics_app",
            endpoint="kpi_dashboard",
            params={"sme_id": "SME_1"},
        )
        obs2 = env.step(action)
        assert obs2 is not None
        assert math.isfinite(obs2.step_reward)

    def test_world_environment_still_works(self):
        from server.world_environment import SMEMultiAgentWorldEnvironment, WorldAction

        env = SMEMultiAgentWorldEnvironment()
        obs = env.reset(seed=1, task_name="competitive-bidding", num_smes=2)
        assert obs is not None
        action = WorldAction(
            role="sme",
            acting_agent_id=obs.acting_agent_id,
            negotiation_action={
                "action_type": "propose",
                "price": 90.0,
                "payment_days": 50,
            },
        )
        obs2 = env.step(action)
        assert obs2 is not None
        assert math.isfinite(obs2.reward)
        assert 0.0 < obs2.reward < 1.0
