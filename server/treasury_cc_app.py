"""FastAPI application for the TreasuryCommandCenter environment."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from openenv.core import create_app
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect
from uvicorn.protocols.utils import ClientDisconnected

from treasury_command_center_env.models import TreasuryCCAction, TreasuryCCObservation

from .concurrency import OpenEnvConcurrencyLimiter, max_concurrent_envs_from_env
from .treasury_cc_environment import TreasuryCommandCenterEnvironment


def _benign_ws_teardown(exc: BaseException) -> bool:
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


def _wrap_ws(app: FastAPI) -> None:
    for route in app.router.routes:
        if isinstance(route, WebSocketRoute) and route.path == "/ws":
            orig: Any = route.endpoint

            async def _safe_ws(ws: WebSocket, *, _orig: Any = orig) -> None:
                try:
                    await _orig(ws)
                except Exception as e:
                    if _benign_ws_teardown(e):
                        return
                    raise

            route.endpoint = _safe_ws
            return


_max = max_concurrent_envs_from_env()

app = create_app(
    TreasuryCommandCenterEnvironment,
    TreasuryCCAction,
    TreasuryCCObservation,
    env_name="treasury-command-center",
    max_concurrent_envs=_max,
)

_wrap_ws(app)
app.add_middleware(OpenEnvConcurrencyLimiter, max_concurrent=_max)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "server.treasury_cc_app:app",
        host="0.0.0.0",
        port=int(os.getenv("TCC_PORT", "7863")),
        reload=False,
    )


if __name__ == "__main__":
    main()
