import numpy as np

from util.msgpack_helper import pack, unpack


def test_roundtrip_keeps_dtype_and_shape():
    message = {"image": np.arange(24, dtype=np.uint8).reshape(2, 4, 3), "state": np.float32([1.5, -2]), "task": "t",
               "nested": {"x": np.int64(3)}}
    decoded = unpack(pack(message))
    assert decoded["image"].dtype == np.uint8 and decoded["image"].shape == (2, 4, 3)
    np.testing.assert_array_equal(decoded["image"], message["image"])
    assert decoded["state"].dtype == np.float32
    assert decoded["task"] == "t" and decoded["nested"]["x"] == 3
