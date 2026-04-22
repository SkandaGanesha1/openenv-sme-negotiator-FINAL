"""OpenEnv server-side modules."""

# Lazy import — avoids pulling in treasury_agent_env (and simpy/networkx)
# when only world_app or sme_app is loaded.
def __getattr__(name: str):
    if name == "TreasuryAgentEnvironment":
        from .treasury_environment import TreasuryAgentEnvironment
        return TreasuryAgentEnvironment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["TreasuryAgentEnvironment"]
