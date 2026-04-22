"""FastAPI application for the TreasuryAgent environment."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from openenv.core import create_app
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect
from uvicorn.protocols.utils import ClientDisconnected

from treasury_agent_env.models import TreasuryAction, TreasuryObservation

from .concurrency import OpenEnvConcurrencyLimiter, max_concurrent_envs_from_env
from .treasury_environment import TreasuryAgentEnvironment


def _benign_websocket_teardown(exc: BaseException) -> bool:
    if isinstance(exc, (WebSocketDisconnect, ClientDisconnected)):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "close message" in msg and "send" in msg:
            return True
    mod = getattr(type(exc), "__module__", "") or ""
    if mod.startswith("websockets.") and "ConnectionClosed" in type(exc).__name__:
        return True
    return False


def _wrap_ws_for_graceful_client_close(app: FastAPI) -> None:
    for route in app.router.routes:
        if isinstance(route, WebSocketRoute) and route.path == "/ws":
            orig: Any = route.endpoint

            async def _safe_ws(websocket: WebSocket, *, _orig: Any = orig) -> None:
                try:
                    await _orig(websocket)
                except Exception as e:
                    if _benign_websocket_teardown(e):
                        return
                    raise

            route.endpoint = _safe_ws
            return


_max = max_concurrent_envs_from_env()

app = create_app(
    TreasuryAgentEnvironment,
    TreasuryAction,
    TreasuryObservation,
    env_name="treasury-agent",
    max_concurrent_envs=_max,
)

_wrap_ws_for_graceful_client_close(app)
app.add_middleware(OpenEnvConcurrencyLimiter, max_concurrent=_max)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "server.treasury_app:app",
        host="0.0.0.0",
        port=int(os.getenv("TREASURY_PORT", "7862")),
        reload=False,
    )


if __name__ == "__main__":
    main()
