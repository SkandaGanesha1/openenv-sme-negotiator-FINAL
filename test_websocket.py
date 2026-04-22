"""Test the multi-agent world server via WebSocket."""
import asyncio
import json
import sys
import websockets


async def run_episode(task_name="competitive-bidding", seed=42):
    uri = "ws://localhost:7861/ws"
    print(f"Connecting to {uri} ...")
    async with websockets.connect(uri) as ws:
        # 1. Reset
        await ws.send(json.dumps({
            "type": "reset",
            "data": {"task_name": task_name, "seed": seed}
        }))
        msg = json.loads(await ws.recv())
        obs = msg["data"]["observation"]
        print(f"Reset OK | Mode: {obs['mode']} | First actor: {obs['acting_agent_id']}")

        # 2. Step loop
        for step in range(200):
            await ws.send(json.dumps({
                "type": "step",
                "data": {
                    "role": "sme",
                    "acting_agent_id": obs["acting_agent_id"],
                    "negotiation_action": {
                        "action_type": "propose",
                        "price": 96.0,
                        "payment_days": 52
                    }
                }
            }))
            msg = json.loads(await ws.recv())
            data = msg["data"]
            obs = data["observation"]
            print(
                f"  Step {step+1}: reward={data['reward']:.4f} "
                f"done={data['done']} actor={obs['acting_agent_id']}"
            )
            if data["done"]:
                print("Episode finished.")
                break


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "competitive-bidding"
    asyncio.run(run_episode(task_name=task))
