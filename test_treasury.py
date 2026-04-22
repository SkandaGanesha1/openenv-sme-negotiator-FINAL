"""Test TreasuryAgent server via WebSocket — all 3 difficulties."""
import asyncio
import json
import websockets

URI = "ws://localhost:7862/ws"


async def run_task(task_name: str, seed: int = 42, max_steps: int = 10):
    async with websockets.connect(URI) as ws:
        # Reset
        await ws.send(json.dumps({"type": "reset", "data": {"task_name": task_name, "seed": seed}}))
        msg = json.loads(await ws.recv())
        obs = msg["data"]["observation"]
        sme = obs["primary_sme_id"]
        print(f"\n{'='*55}")
        print(f"Task : {task_name}")
        print(f"SME  : {sme}  |  Day 1/{obs['max_days']}  |  Cash buffer: {obs['cash_buffer_days']:.1f}d")
        print(f"Overdraft limit: {obs['overdraft_limit']:,.0f}  |  Solvency: {obs['solvency_ok']}")

        # Cycle through all 6 tool apps
        tools = [
            ("erp_app",        "invoice_summary",     {"sme_id": sme, "window_days": 30}),
            ("bank_app",       "get_balances",        {"sme_id": sme}),
            ("treds_app",      "eligibility_summary", {"sme_id": sme}),
            ("dd_app",         "propose_discount_scheme", {"buyer_id": "BUYER_A", "target_days": 45, "max_discount_pct": 2.0}),
            ("compliance_app", "check_45_day_breach", {"sme_id": sme, "buyer_id": "BUYER_A"}),
            ("analytics_app",  "kpi_dashboard",       {"sme_id": sme}),
        ]

        done = False
        for step in range(max_steps):
            app, endpoint, params = tools[step % len(tools)]
            await ws.send(json.dumps({"type": "step", "data": {
                "role": "treasury",
                "command_type": "tool_call",
                "app": app,
                "endpoint": endpoint,
                "params": params,
            }}))
            msg = json.loads(await ws.recv())
            d = msg["data"]
            obs = d["observation"]
            result = obs.get("last_tool_result", {})
            status = "ok" if "error" not in result else f"ERR: {result['error']}"
            print(f"  Step {step+1:>2}: {app}.{endpoint:<30} reward={d['reward']:.4f}  done={d['done']}  [{status}]")
            if d["done"]:
                done = True
                break

        if not done:
            print(f"  ... episode still running (ran {max_steps} steps)")


async def main():
    print("Connecting to TreasuryAgent at", URI)
    await run_task("treasury-easy",   seed=42, max_steps=6)
    await run_task("treasury-medium", seed=99, max_steps=8)
    await run_task("treasury-hard",   seed=7,  max_steps=10)
    print("\nAll tasks completed successfully.")


asyncio.run(main())
