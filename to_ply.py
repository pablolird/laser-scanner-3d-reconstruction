#!/usr/bin/env python3
"""Convert output.xyz (X Y Z R G B) to a colored PLY readable by MeshLab."""
import numpy as np, struct, os
from pathlib import Path

src = Path("output.xyz")
dst = src.with_suffix(".ply")

pts = np.loadtxt(src)
n = len(pts)

header = (
    "ply\n"
    "format binary_little_endian 1.0\n"
    f"element vertex {n}\n"
    "property float x\n"
    "property float y\n"
    "property float z\n"
    "property uchar red\n"
    "property uchar green\n"
    "property uchar blue\n"
    "end_header\n"
).encode()

with open(dst, "wb") as f:
    f.write(header)
    for row in pts:
        x, y, z = float(row[0]), float(row[1]), float(row[2])
        r, g, b = int(row[3]), int(row[4]), int(row[5])
        f.write(struct.pack("<fffBBB", x, y, z, r, g, b))

print(f"Saved {dst}  ({n} points)")
