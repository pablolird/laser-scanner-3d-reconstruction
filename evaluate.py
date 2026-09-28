#!/usr/bin/env python3
"""
Accuracy of a reconstructed point cloud against the ground-truth mesh.

For every reconstructed point, computes the distance to the closest triangle
of data/ground_truth.ply (point-to-surface, like CloudCompare's cloud-to-mesh).
Candidate triangles are those touching the K nearest mesh vertices, which is
exact for a reasonably uniform mesh and never underestimates the error.

Usage:
    python evaluate.py                       # evaluates output.xyz
    python evaluate.py --cloud my_cloud.xyz --gnome-only
"""

import argparse
import struct
import numpy as np
from scipy.spatial import cKDTree

PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8",
             "uchar": "u1", "uint8": "u1", "int": "<i4", "uint": "<u4"}


def load_mesh(path):
    """Minimal binary-little-endian PLY reader: vertices (V,3) and triangles (F,3). Quads are split."""
    with open(path, "rb") as f:
        header = []
        while (line := f.readline().decode().strip()) != "end_header":
            header.append(line)
        body = f.read()

    n_vert = n_face = 0
    vprops, element = [], None
    for line in header:
        tok = line.split()
        if tok[0] == "element":
            element = tok[1]
            if element == "vertex": n_vert = int(tok[2])
            if element == "face":   n_face = int(tok[2])
        elif tok[0] == "property" and element == "vertex":
            vprops.append((tok[2], PLY_TYPES[tok[1]]))

    vdt = np.dtype(vprops)
    v = np.frombuffer(body, dtype=vdt, count=n_vert)
    V = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)

    # faces: variable-length lists (uchar count, uint indices)
    tris, o = [], n_vert * vdt.itemsize
    for _ in range(n_face):
        n = body[o]
        ids = struct.unpack_from(f"<{n}I", body, o + 1)
        o += 1 + 4 * n
        tris.extend((ids[0], ids[k], ids[k + 1]) for k in range(1, n - 1))
    return V, np.array(tris, dtype=np.int64)


def point_triangle_dist(P, A, B, C):
    """Vectorised closest-point-on-triangle distance (Ericson, Real-Time Collision Detection 5.1.5)."""
    eps = 1e-12
    AB, AC, AP = B - A, C - A, P - A
    d1, d2 = (AB * AP).sum(-1), (AC * AP).sum(-1)
    BP = P - B; d3, d4 = (AB * BP).sum(-1), (AC * BP).sum(-1)
    CP = P - C; d5, d6 = (AB * CP).sum(-1), (AC * CP).sum(-1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2

    # default: projection falls inside the face
    den = va + vb + vc; den = np.where(den == 0, eps, den)
    Q = A + AB * (vb / den)[:, None] + AC * (vc / den)[:, None]

    # edge regions
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = d1 / np.where(d1 - d3 == 0, eps, d1 - d3); Q[m] = (A + AB * t[:, None])[m]
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = d2 / np.where(d2 - d6 == 0, eps, d2 - d6); Q[m] = (A + AC * t[:, None])[m]
    e1, e2 = d4 - d3, d5 - d6
    m = (va <= 0) & (e1 >= 0) & (e2 >= 0)
    t = e1 / np.where(e1 + e2 == 0, eps, e1 + e2); Q[m] = (B + (C - B) * t[:, None])[m]

    # vertex regions
    m = (d1 <= 0) & (d2 <= 0);  Q[m] = A[m]
    m = (d3 >= 0) & (d4 <= d3); Q[m] = B[m]
    m = (d6 >= 0) & (d5 <= d6); Q[m] = C[m]
    return np.linalg.norm(P - Q, axis=1)


def cloud_to_mesh(P, V, F, k=8):
    """Distance from each point in P to the nearest triangle among those touching its k nearest vertices."""
    # vertex -> incident faces, padded with -1
    deg = np.bincount(F.ravel(), minlength=len(V))
    vf = np.full((len(V), deg.max()), -1, dtype=np.int64)
    fill = np.zeros(len(V), dtype=np.int64)
    for fi, tri in enumerate(F):
        for vi in tri:
            vf[vi, fill[vi]] = fi; fill[vi] += 1

    _, nn = cKDTree(V).query(P, k=k)
    best = np.full(len(P), np.inf)
    for col in range(k):
        cand = vf[nn[:, col]]                       # (N, max_deg)
        for s in range(cand.shape[1]):
            ok = cand[:, s] >= 0
            f = F[cand[ok, s]]
            d = point_triangle_dist(P[ok], V[f[:, 0]], V[f[:, 1]], V[f[:, 2]])
            best[ok] = np.minimum(best[ok], d)
    return best


def summarise(label, d):
    pct = lambda t: 100.0 * (d < t).mean()
    print(f"{label}  ({len(d):,} points)")
    print(f"  mean {d.mean():.2f} mm   median {np.median(d):.2f} mm   std {d.std():.2f} mm")
    print(f"  90th pct {np.percentile(d, 90):.2f} mm   95th pct {np.percentile(d, 95):.2f} mm")
    print(f"  within 1 mm: {pct(1):.1f}%   2 mm: {pct(2):.1f}%   5 mm: {pct(5):.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cloud", default="output.xyz", help="reconstructed X Y Z R G B file")
    ap.add_argument("--gt", default="data/ground_truth.ply", help="ground-truth mesh (binary PLY)")
    ap.add_argument("--gnome-only", action="store_true",
                    help="also report points above the turntable top (Z > 0.5 mm) separately")
    args = ap.parse_args()

    V, F = load_mesh(args.gt)
    P = np.loadtxt(args.cloud, usecols=(0, 1, 2))
    print(f"Ground truth: {len(V):,} vertices, {len(F):,} triangles, "
          f"height {np.ptp(V[:, 2]):.1f} mm")
    print(f"Cloud Z range: {P[:, 2].min():.1f} to {P[:, 2].max():.1f} mm "
          f"(ground truth {V[:, 2].min():.1f} to {V[:, 2].max():.1f} mm)\n")

    d = cloud_to_mesh(P, V, F)
    summarise("Cloud -> mesh distance, all points", d)
    if args.gnome_only:
        m = P[:, 2] > 0.5
        summarise("\nCloud -> mesh distance, figurine only (Z > 0.5 mm)", d[m])
