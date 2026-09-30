"""策略服务的消息编码：msgpack + numpy 数组（dtype、shape、原始字节）。"""

import msgpack
import numpy as np
from loguru import logger


def encode_default(obj):
    if isinstance(obj, np.ndarray):
        array = np.ascontiguousarray(obj)
        result = {"__ndarray__": True, "dtype": array.dtype.str, "shape": list(array.shape), "data": array.tobytes()}
        return result
    if isinstance(obj, np.generic):
        result = obj.item()
        return result
    raise TypeError(f"不能编码 {type(obj)}")


def decode_hook(obj):
    if obj.get("__ndarray__"):
        result = np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
        return result
    return obj


def pack(message):
    data = msgpack.packb(message, default=encode_default, use_bin_type=True)
    return data


def unpack(data):
    message = msgpack.unpackb(data, object_hook=decode_hook, raw=False)
    return message


def demo_roundtrip():
    message = {"image": np.zeros((480, 640, 3), dtype=np.uint8), "state": np.arange(16, dtype=np.float32)}
    data = pack(message)
    decoded = unpack(data)
    logger.info("{} bytes; state equal: {}", len(data), np.array_equal(decoded["state"], message["state"]))


def main():
    demo_roundtrip()


if __name__ == "__main__":
    main()
