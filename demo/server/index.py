import asyncio

import websockets


async def handler(websocket, path):
    try:
        async for message in websocket:
            print(f"Received message: {message}")
            await websocket.send(f"Echo: {message}")
    except websockets.ConnectionClosedOK:
        print("Connection closed")


async def main():
    server = await websockets.serve(handler, "localhost", 8000)
    print("WebSocket server is running on ws://localhost:8000")
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
