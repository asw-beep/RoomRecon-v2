"""Minimal glTF 2.0 binary writer for the collision tests: one mesh, one triangle primitive,
optional node translation."""
import json
import struct

import numpy as np

_INDEX_DTYPE = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32}


def _pad(b: bytes, fill: bytes) -> bytes:
    return b + fill * ((4 - len(b) % 4) % 4)


def write_glb(path, V, F, index_type=5125, translation=None):
    v = np.asarray(V, np.float32)
    i = np.asarray(F).astype(_INDEX_DTYPE[index_type]).reshape(-1)
    vb = _pad(v.tobytes(), b"\0")
    ib = _pad(i.tobytes(), b"\0")
    node = {"mesh": 0}
    if translation is not None:
        node["translation"] = [float(x) for x in translation]
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [node],
           "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
           "buffers": [{"byteLength": len(vb) + len(ib)}],
           "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": v.nbytes},
                           {"buffer": 0, "byteOffset": len(vb), "byteLength": i.nbytes}],
           "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(v), "type": "VEC3",
                          "min": v.min(0).tolist(), "max": v.max(0).tolist()},
                         {"bufferView": 1, "componentType": index_type, "count": len(i), "type": "SCALAR"}]}
    js = _pad(json.dumps(doc).encode(), b" ")
    bin_ = vb + ib
    total = 12 + 8 + len(js) + 8 + len(bin_)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(bin_), 0x004E4942) + bin_)
