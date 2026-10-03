#!/usr/bin/env python
"""Write a 3-splat PLY at (1,0,0), (0,1,0), (0,0,1) to learn splat-transform's Euler convention.
Usage: python scripts/make_test_ply.py out.ply"""
import sys

import numpy as np

props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
         "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
rows = np.zeros((3, len(props)), dtype=np.float32)
rows[:, :3] = np.eye(3, dtype=np.float32)
rows[:, 6] = 5.0
rows[:, 7:10] = -4.0
rows[:, 10] = 1.0
header = ("ply\nformat binary_little_endian 1.0\nelement vertex 3\n"
          + "".join(f"property float {p}\n" for p in props) + "end_header\n")
with open(sys.argv[1], "wb") as f:
    f.write(header.encode())
    f.write(rows.tobytes())
