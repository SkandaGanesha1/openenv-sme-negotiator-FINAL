"""
TreasuryActionRouter — dispatches a TreasuryAction JSON to the correct
simulated tool app and returns the structured result dict.

Design contract:
  - Every tool app class receives (world, rng, belief) as needed.
  - Endpoint names are validated against each app's public methods.
  - Unknown app or endpoint → structured error (never raises).
  - params dict is unpacked as kwargs into the endpoint method.
"""

from __future__ import annotations

from random import Random
from typing import Any

from .models import TreasuryAction
from .world_state import TreasuryWorldState
from .belief_state import TreasuryBeliefState
from .tools.erp_app import ErpApp
from .tools.bank_app import BankApp
from .tools.treds_app import TredsApp
from .tools.dd_app import DdApp
from .tools.compliance_app import ComplianceApp
from .tools.analytics_app import AnalyticsApp

# Allowed endpoints per app (whitelist — prevents arbitrary method calls)
_ALLOWED_ENDPOINTS: dict[str, frozenset[str]] = {
    "erp_app": frozenset(
        {"list_invoices", "invoice_summary", "projected_cashflow", "update_terms"}
    ),
    "bank_app": frozenset(
        {"get_balances", "draw_overdraft", "repay_overdraft", "view_covenants"}
    ),
    "treds_app": frozenset(
        {"quote_discount_rate", "discount_invoice", "eligibility_summary"}
    ),
    "dd_app": frozenset(
        {"propose_discount_scheme", "simulate_scheme", "activate_scheme"}
    ),
    "compliance_app": frozenset(
        {"check_45_day_breach", "estimate_43B_tax_impact", "prepare_samadhaan_case"}
    ),
    "analytics_app": frozenset(
        {"portfolio_risks", "scenario_analysis", "kpi_dashboard"}
    ),
}


class TreasuryActionRouter:
    """Routes one TreasuryAction to the appropriate tool app instance."""

    def __init__(
        self,
        world: TreasuryWorldState,
        belief: TreasuryBeliefState,
        rng: Random,
    ) -> None:
        self._world = world
        self._belief = belief
        self._rng = rng

        # Instantiate all apps once (stateless wrappers around world)
        self._apps: dict[str, Any] = {
            "erp_app": ErpApp(world),
            "bank_app": BankApp(world),
            "treds_app": TredsApp(world, rng),
            "dd_app": DdApp(world, rng),
            "compliance_app": ComplianceApp(world),
            "analytics_app": AnalyticsApp(world, belief),
        }

    def route(self, action: TreasuryAction) -> dict[str, Any]:
        """
        Execute the tool call described by `action`.

        Returns a result dict always containing "app" and "endpoint" keys.
        Never raises — errors are returned as {"error": "..."} dicts.
        """
        app_name = action.app
        endpoint = action.endpoint
        params = action.params or {}

        # Validate app
        if app_name not in self._apps:
            return {
                "error": f"Unknown app: '{app_name}'. "
                         f"Valid apps: {sorted(self._apps)}",
                "app": app_name,
                "endpoint": endpoint,
            }

        # Validate endpoint whitelist
        allowed = _ALLOWED_ENDPOINTS.get(app_name, frozenset())
        if endpoint not in allowed:
            return {
                "error": f"Unknown endpoint: '{endpoint}' for app '{app_name}'. "
                         f"Allowed: {sorted(allowed)}",
                "app": app_name,
                "endpoint": endpoint,
            }

        # Dispatch
        app_instance = self._apps[app_name]
        method = getattr(app_instance, endpoint)
        try:
            result = method(**params)
        except TypeError as exc:
            return {
                "error": f"Bad params for {app_name}.{endpoint}: {exc}",
                "app": app_name,
                "endpoint": endpoint,
            }
        except Exception as exc:
            return {
                "error": f"Tool execution error in {app_name}.{endpoint}: {exc}",
                "app": app_name,
                "endpoint": endpoint,
            }

        # Ensure result always has app/endpoint labels
        if isinstance(result, dict):
            result.setdefault("app", app_name)
            result.setdefault("endpoint", endpoint)
        return result

    def is_observational(self, action: TreasuryAction) -> bool:
        """True when the action only reads state (no side-effects on world)."""
        read_only = {
            "erp_app": {"list_invoices", "invoice_summary", "projected_cashflow"},
            "bank_app": {"get_balances", "view_covenants"},
            "treds_app": {"quote_discount_rate", "eligibility_summary"},
            "dd_app": {"propose_discount_scheme", "simulate_scheme"},
            "compliance_app": {"check_45_day_breach", "estimate_43B_tax_impact"},
            "analytics_app": {"portfolio_risks", "scenario_analysis", "kpi_dashboard"},
        }
        return action.endpoint in read_only.get(action.app, set())
