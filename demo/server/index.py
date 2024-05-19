import argparse
import asyncio
import json
import os
import sys

import torch
import websockets

sys.path.append("../../")

from tiny import PointCloudDiffusion
from tiny.utils import plot_point_clouds

store = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="../../weights/modelnet40")

    return parser.parse_args()


def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


async def handler(websocket, path):
    try:
        async for message in websocket:
            print(f"Received message: {message}")
            message = json.loads(message)
            prompt = message["prompt"]
            diffusion = store["diffusion"]

            for sample in diffusion.sample_loop_progressive(
                1, [prompt], num_inference_steps=20, guidance_scale=2.0, use_cfg=True
            ):
                points = sample[0].tolist()
                data = json.dumps({"points": points})
                await websocket.send(data)

    except websockets.ConnectionClosedOK:
        print("Connection closed")


async def main():
    # load model
    args = parse_args()
    config_path = os.path.join(args.weights, "config.json")
    weights_path = os.path.join(args.weights, "weights.pt")
    config = read_json(config_path)
    checkpoint = torch.load(weights_path)

    if "weights" in checkpoint:
        checkpoint = checkpoint["weights"]

    diffusion = PointCloudDiffusion.from_config(config)
    diffusion.model.load_state_dict(checkpoint)
    store["diffusion"] = diffusion

    # start server
    server = await websockets.serve(handler, "localhost", 8000)
    print("WebSocket server is running on ws://localhost:8000")
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
