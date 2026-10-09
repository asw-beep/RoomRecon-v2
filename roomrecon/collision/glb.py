"""Read the triangles of a glTF 2.0 binary (`splat-transform --collision-mesh` writes one) and
sample points on them. Stdlib + numpy only (no trimesh on the PC)."""
import json
import struct
from pathlib import Path

import numpy as np

_DTYPE = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _accessor(doc, binary, i) -> np.ndarray:
    acc = doc["accessors"][i]
    view = doc["bufferViews"][acc["bufferView"]]
    dtype, width = np.dtype(_DTYPE[acc["componentType"]]), _WIDTH[acc["type"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = view.get("byteStride", dtype.itemsize * width)
    rows = np.ndarray((acc["count"], width), dtype, binary, start, (stride, dtype.itemsize))
    return rows.copy()


def _local(node) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], float).reshape(4, 4).T          # glTF stores column-major
    M = np.eye(4)
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M[:3, :3] = R * np.array(node.get("scale", [1, 1, 1]))
    M[:3, 3] = node.get("translation", [0, 0, 0])
    return M


def read_glb(path) -> tuple[np.ndarray, np.ndarray]:
    raw = Path(path).read_bytes()
    magic, version, _ = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError(f"{path} is not a glTF 2.0 binary")
    off, doc, binary = 12, None, b""
    while off < len(raw):
        length, kind = struct.unpack_from("<II", raw, off)
        chunk = raw[off + 8: off + 8 + length]
        if kind == 0x4E4F534A:
            doc = json.loads(chunk)
        elif kind == 0x004E4942:
            binary = chunk
        off += 8 + length
    verts, faces, base = [], [], 0

    def visit(i, parent):
        nonlocal base
        node = doc["nodes"][i]
        M = parent @ _local(node)
        for prim in doc["meshes"][node["mesh"]]["primitives"] if "mesh" in node else []:
            if prim.get("mode", 4) != 4:
                continue
            v = _accessor(doc, binary, prim["attributes"]["POSITION"]).astype(float)
            f = (_accessor(doc, binary, prim["indices"]).reshape(-1, 3).astype(np.int64)
                 if "indices" in prim else np.arange(len(v)).reshape(-1, 3))
            verts.append(v @ M[:3, :3].T + M[:3, 3])
            faces.append(f + base)
            base += len(v)
        for c in node.get("children", []):
            visit(c, M)

    scene = doc.get("scenes", [{"nodes": list(range(len(doc.get("nodes", []))))}])[doc.get("scene", 0)]
    for i in scene["nodes"]:
        visit(i, np.eye(4))
    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    return np.vstack(verts), np.vstack(faces)


def sample_triangles(V, F, spacing=0.01, seed=0, cap=4_000_000) -> np.ndarray:
    """Uniform samples over the triangles' area, about one per spacing² of surface."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = area.sum()
    if total <= 0:
        return np.zeros((0, 3))
    n = int(min(cap, max(1, round(total / spacing ** 2))))
    rng = np.random.default_rng(seed)
    t = rng.choice(len(F), n, p=area / total)
    r1, r2 = np.sqrt(rng.random(n))[:, None], rng.random(n)[:, None]
    return (1 - r1) * a[t] + r1 * (1 - r2) * b[t] + r1 * r2 * c[t]
