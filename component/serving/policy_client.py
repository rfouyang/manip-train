"""机器人端客户端（同步）。只依赖 numpy、msgpack、websockets，可以直接拷进 ROS 2 节点使用。"""

import time

import numpy as np
from loguru import logger

from util.msgpack_helper import pack, unpack


class PolicyClient:
    def __init__(self, url="ws://127.0.0.1:8765", timeout=10.0):
        from websockets.sync.client import connect

        self.connection = connect(url, max_size=None, compression=None, open_timeout=timeout)
        self.manifest = unpack(self.connection.recv())["manifest"]

    def request(self, message):
        self.connection.send(pack(message))
        reply = unpack(self.connection.recv())
        if "error" in reply:
            raise RuntimeError(f"策略服务报错：{reply['error']}")
        return reply

    def reset(self):
        self.request({"type": "reset"})

    def infer(self, observation):
        reply = self.request({"type": "infer", "observation": observation})
        return reply

    def close(self):
        self.connection.close()


def demo_client(url="ws://127.0.0.1:8765", steps=50):
    """先运行 component/serving/policy_server.py，再运行本文件。"""
    from component.serving.policy_runtime import dummy_observation

    client = PolicyClient(url)
    logger.info("Connected: {} ({})", client.manifest["name"], client.manifest["policy"])
    client.reset()
    observation = dummy_observation(client.manifest)
    round_trip = []
    for _ in range(steps):
        started = time.perf_counter()
        reply = client.infer(observation)
        round_trip.append((time.perf_counter() - started) * 1000)
    logger.info("Action dims {}; round trip mean {:.2f} ms, p95 {:.2f} ms", reply["action"].shape,
                np.mean(round_trip[1:]), np.percentile(round_trip[1:], 95))
    client.close()


def main():
    demo_client()


if __name__ == "__main__":
    main()
