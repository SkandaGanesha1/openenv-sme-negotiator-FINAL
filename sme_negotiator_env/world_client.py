"""Python client for the SME multi-agent world environment.

Provides a simple synchronous-friendly async interface compatible with
TRL / Unsloth GRPOTrainer via openenv.core.GenericEnvClient.

Example usage:
    import asyncio
    from sme_negotiator_env.world_client import SMEWorldEnvClient, build_sme_action

    async def run():
        async with SMEWorldEnvClient("http://localhost:7861") as client:
            obs = await client.reset(task_name="competitive-bidding", num_smes=3)
            print(obs.text)

            action = build_sme_action(
                acting_agent_id=obs.acting_agent_id,
                price=95.0,
                payment_days=45,
                reason="Opening anchor below buyer ask",
            )
            result = await client.step(action)
            print(f"Reward: {result.reward}, Done: {result.done}")
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from openenv.core import GenericEnvClient
from openenv.core.client_types import StepResult

from server.world_environment import WorldAction, WorldObservation


class SMEWorldEnvClient(GenericEnvClient):
    """Typed OpenEnv client for the multi-agent world environment."""

    def __init__(
        self,
        base_url: str,
        connect_timeout_s: float = 10.0,
        message_timeout_s: float = 60.0,
        max_message_size_mb: float = 100.0,
        provider: Optional[Any] = None,
        mode: Optional[str] = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            connect_timeout_s=connect_timeout_s,
            message_timeout_s=message_timeout_s,
            max_message_size_mb=max_message_size_mb,
            provider=provider,
            mode=mode,
        )
        self._last_observation: Optional[WorldObservation] = None

    def _step_payload(self, action: WorldAction) -> Dict[str, Any]:
        if hasattr(action, "model_dump"):
            return action.model_dump()
        return dict(action)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[WorldObservation]:
        obs_payload = payload.get("observation", payload)
        if isinstance(obs_payload, WorldObservation):
            obs = obs_payload
        else:
            obs = WorldObservation(**obs_payload)
        self._last_observation = obs
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return dict(payload)

    @property
    def last_observation(self) -> Optional[WorldObservation]:
        return self._last_observation


# ---------------------------------------------------------------------------
# Action factory helpers (for training scripts and notebooks)
# ---------------------------------------------------------------------------


def build_sme_action(
    *,
    acting_agent_id: str,
    action_type: str = "propose",
    price: float,
    payment_days: int,
    use_treds: bool = False,
    reason: str = "",
    propose_late_payment_penalty_clause: bool = False,
    propose_dynamic_discounting: bool = False,
    dynamic_discount_annual_rate: float = 0.0,
    coalition_message: Optional[str] = None,
) -> WorldAction:
    """Build a WorldAction for an SME agent."""
    return WorldAction(
        role="sme",
        acting_agent_id=acting_agent_id,
        negotiation_action={
            "action_type": action_type,
            "price": price,
            "payment_days": payment_days,
            "use_treds": use_treds,
            "reason": reason,
            "propose_late_payment_penalty_clause": propose_late_payment_penalty_clause,
            "propose_dynamic_discounting": propose_dynamic_discounting,
            "dynamic_discount_annual_rate": dynamic_discount_annual_rate,
        },
        coalition_message=coalition_message,
    )


def build_oversight_action(
    *,
    flag_unfair_cases: Optional[list] = None,
    suggested_interventions: Optional[Dict[str, str]] = None,
    global_explanation: str = "",
) -> WorldAction:
    """Build a WorldAction for the OversightAgent (Mode C)."""
    return WorldAction(
        role="oversight",
        acting_agent_id="oversight_agent",
        flag_unfair_cases=flag_unfair_cases or [],
        suggested_interventions=suggested_interventions or {},
        global_explanation=global_explanation,
    )


def build_manager_action(
    *,
    instructions: Optional[Dict[str, str]] = None,
    query_tool: Optional[str] = None,
) -> WorldAction:
    """Build a WorldAction for the ManagerAgent (Mode D)."""
    return WorldAction(
        role="manager",
        acting_agent_id="manager_agent",
        instructions=instructions or {},
        query_tool=query_tool,
    )
