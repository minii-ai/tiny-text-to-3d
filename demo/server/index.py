import argparse
import asyncio
import json
import os
import sys

import torch
import websockets

sys.path.append("../../")

from tiny import PointCloudDiffusion

store = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="../../weights/modelnet40")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--use_cfg", type=bool, default=True)

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

            i = 0
            num_inference_steps = store["num_inference_steps"]
            guidance_scale = store["guidance_scale"]
            use_cfg = store["use_cfg"]

            for sample in diffusion.sample_loop_progressive(
                1,
                [prompt],
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                use_cfg=use_cfg,
            ):
                progress = (i + 1) / num_inference_steps
                points = sample[0].tolist()
                data = json.dumps({"points": points, "progress": progress})
                await websocket.send(data)
                i += 1

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

    # save inference configs
    store["num_inference_steps"] = args.num_inference_steps
    store["guidance_scale"] = args.guidance_scale
    store["use_cfg"] = args.use_cfg

    # start server
    server = await websockets.serve(handler, "localhost", args.port)
    print(f"WebSocket server is running on ws://localhost:{args.port}")
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
