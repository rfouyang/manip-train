"""策略服务：WebSocket + msgpack。协议与 openpi 的 serve_policy 相近，机器人端（ROS 2 节点）只需 policy_client。

连接后服务端先发一条元数据（导出包 manifest）；之后客户端每条消息为
{"type": "infer", "observation": {...}} → {"action": ndarray, "latency_ms": float}
{"type": "reset"} → {"ok": true}
出错时回复 {"error": 原因}，连接保持。
"""

import asyncio
import threading

from loguru import logger

from util.msgpack_helper import pack, unpack


class PolicyServer:
    def __init__(self, runtime, host="0.0.0.0", port=8765):
        self.runtime = runtime
        self.host = host
        self.port = port
        # 一个 GPU 模型：同一时刻只推理一个请求，多个客户端按到达顺序排队。
        self.lock = threading.Lock()

    def handle_message(self, message):
        kind = message.get("type")
        with self.lock:
            if kind == "reset":
                self.runtime.reset()
                result = {"ok": True}
            elif kind == "infer":
                result = self.runtime.infer(message["observation"])
            else:
                result = {"error": f"未知消息类型：{kind}"}
        return result

    async def handler(self, websocket):
        logger.info("Client connected: {}", websocket.remote_address)
        await websocket.send(pack({"type": "metadata", "manifest": self.runtime.manifest}))
        async for data in websocket:
            try:
                result = await asyncio.to_thread(self.handle_message, unpack(data))
            except (KeyError, ValueError) as error:
                logger.warning("Bad request: {}", error)
                result = {"error": str(error)}
            await websocket.send(pack(result))
        logger.info("Client disconnected: {}", websocket.remote_address)

    async def serve_forever(self, ready=None):
        from websockets.asyncio.server import serve

        async with serve(self.handler, self.host, self.port, max_size=None, compression=None) as server:
            logger.info("Policy server on ws://{}:{}", self.host, self.port)
            if ready is not None:
                ready.set()
            await server.serve_forever()

    def run(self):
        asyncio.run(self.serve_forever())


def main():
    from pathlib import Path

    from component.serving.policy_runtime import PolicyRuntime

    root = Path(__file__).resolve().parents[2]
    bundles = sorted((root / "output").glob("**/models/*/manifest.json"))
    if not bundles:
        logger.info("No bundle yet; run component/training/train_runner.py first")
        return
    PolicyServer(PolicyRuntime(bundles[-1].parent), host="127.0.0.1").run()


if __name__ == "__main__":
    main()
