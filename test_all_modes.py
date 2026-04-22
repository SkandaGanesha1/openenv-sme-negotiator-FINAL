"""Test all 4 multi-agent modes via WebSocket (stateful sessions)."""
import asyncio
import json
import sys
import websockets


async def ws_reset(ws, task_name, seed):
    await ws.send(json.dumps({"type": "reset", "data": {"task_name": task_name, "seed": seed}}))
    msg = json.loads(await ws.recv())
    return msg["data"]["observation"]


async def ws_step(ws, action_data):
    await ws.send(json.dumps({"type": "step", "data": action_data}))
    msg = json.loads(await ws.recv())
    return msg["data"]


async def mode_a():
    print("=== Mode A: Competitive Bidding ===")
    async with websockets.connect("ws://localhost:7861/ws") as ws:
        obs = await ws_reset(ws, "competitive-bidding", seed=42)
        print(f"SMEs: {obs['aux']['num_smes']} | Buyers: {obs['aux']['num_buyers']}")
        print(f"First actor: {obs['acting_agent_id']}")

        for step in range(5):
            data = await ws_step(ws, {
                "role": "sme",
                "acting_agent_id": obs["acting_agent_id"],
                "negotiation_action": {"action_type": "propose", "price": 96.0, "payment_days": 52}
            })
            obs = data["observation"]
            print(f"  Step {step+1}: reward={data['reward']:.4f} done={data['done']} actor={obs['acting_agent_id']}")
            if data["done"]:
                break
    print()


async def mode_b():
    print("=== Mode B: Coalition Bargaining ===")
    async with websockets.connect("ws://localhost:7861/ws") as ws:
        obs = await ws_reset(ws, "coalition-bargaining", seed=10)
        print(f"SMEs: {obs['aux']['num_smes']} | Pairs: {obs['aux']['pairs_total']}")
        print(f"First actor: {obs['acting_agent_id']}")

        for step in range(5):
            data = await ws_step(ws, {
                "role": "sme",
                "acting_agent_id": obs["acting_agent_id"],
                "coalition_message": "All hold at 50 days — agreed?",
                "negotiation_action": {"action_type": "propose", "price": 95.0, "payment_days": 50}
            })
            obs = data["observation"]
            print(f"  Step {step+1}: reward={data['reward']:.4f} done={data['done']} actor={obs['acting_agent_id']}")
            if data["done"]:
                break
    print()


async def mode_c():
    print("=== Mode C: Oversight Arena ===")
    async with websockets.connect("ws://localhost:7861/ws") as ws:
        obs = await ws_reset(ws, "oversight-arena", seed=7)
        print(f"Role: {obs['role']} | Pairs to watch: {obs['aux']['pairs_total']}")

        for step in range(3):
            data = await ws_step(ws, {
                "role": "oversight",
                "acting_agent_id": "oversight_agent",
                "flag_unfair_cases": ["sme_100_x_buyer_100"],
                "suggested_interventions": {"sme_100_x_buyer_100": "Buyer power too high"},
                "global_explanation": "Buyer power >0.8 with large liquidity gap is exploitative"
            })
            obs = data["observation"]
            print(f"  Step {step+1}: reward={data['reward']:.4f} done={data['done']}")
            print(f"    Ground truth unfair: {obs['aux'].get('ground_truth_unfair_pairs')}")
            print(f"    Precision/Recall: {obs['aux'].get('oversight_precision_recall')}")
            if data["done"]:
                break
    print()


async def mode_d():
    print("=== Mode D: Manager Orchestration ===")
    async with websockets.connect("ws://localhost:7861/ws") as ws:
        obs = await ws_reset(ws, "manager-orchestration", seed=3)
        print(f"Role: {obs['role']} | SMEs to manage: {obs['aux']['num_smes']}")

        for step in range(3):
            data = await ws_step(ws, {
                "role": "manager",
                "acting_agent_id": "manager_agent",
                "instructions": {
                    "sme_0": "propose days=55 price=96.0 use treds",
                    "sme_1": "propose days=50 price=97.0",
                    "sme_2": "propose days=60 price=95.5"
                },
                "query_tool": "treds_rate"
            })
            obs = data["observation"]
            print(f"  Step {step+1}: reward={data['reward']:.4f} done={data['done']}")
            print(f"    Solvent fraction: {obs['aux'].get('solvent_fraction')}")
            print(f"    Gini (fairness): {round(obs['aux'].get('gini_days', 0), 4)}")
            if data["done"]:
                break
    print()


async def main():
    await mode_a()
    await mode_b()
    await mode_c()
    await mode_d()
    print("All 4 modes passed.")


asyncio.run(main())
