#!/usr/bin/env python3
"""
Debug tools for the laser-triangulation pipeline.
Run each tool independently, e.g.:
    conda run -n cv python debug_tools.py laser   [--frame 0]
    conda run -n cv python debug_tools.py calib
    conda run -n cv python debug_tools.py slice   [--frame 0]
    conda run -n cv python debug_tools.py cloud
"""

import cv2, numpy as np, argparse, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DEBUG_DIR = Path("debug")

def _subdir(name):
    d = DEBUG_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d

# ── import calibration/detection helpers from main script ──────────────
sys.path.insert(0, str(Path(__file__).parent))
from reconstruct import (
    calibrate, detect_laser, triangulate,
    rotation_z, K, K_INV, DATA, N_FRAMES
)

# ───────────────────────────────────────────────────────────────────────
# TOOL 1 — laser detection overlay
# Shows the raw frame, no-laser frame, their difference, and the fitted
# laser centroid line, all in one image.
# ───────────────────────────────────────────────────────────────────────
def tool_laser(frame_idx=0, peak_fraction=0.5, min_peak=20):
    laser_img = cv2.imread(str(DATA / f"{frame_idx:04d}.jpg"))
    color_img = cv2.imread(str(DATA / "withoutLaser" / f"{frame_idx:04d}.jpg"))
    assert laser_img is not None and color_img is not None

    diff = laser_img.astype(np.int32) - color_img.astype(np.int32)
    green_diff = np.clip(diff[:, :, 1], 0, 255).astype(np.uint8)

    # Build a visualisation of the peak-filtered mask
    peak_mask = np.zeros_like(green_diff)
    for col in range(laser_img.shape[1]):
        col_diff = green_diff[:, col].astype(int)
        peak = col_diff.max()
        if peak < min_peak:
            continue
        cutoff = peak * peak_fraction
        peak_mask[col_diff >= cutoff, col] = col_diff[col_diff >= cutoff]

    centroids = detect_laser(laser_img, color_img,
                             peak_fraction=peak_fraction, min_peak=min_peak)

    # Draw detected centroid line on a copy of the laser frame
    overlay = laser_img.copy()
    col_layer = np.zeros_like(overlay)
    for col, row in centroids:
        col_layer[int(round(row)), :] = (0, 255, 255)   # yellow row highlight (BGR)
    overlay = cv2.addWeighted(overlay, 1.0, col_layer, 0.12, 0)
    for col, row in centroids:
        cv2.circle(overlay, (int(round(col)), int(round(row))), 3, (0, 0, 255), -1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 8))
    titles = ["Laser frame", "Green diff (raw)",
              f"Peak mask (frac={peak_fraction}, min={min_peak})",
              "Detected points (red) + row coverage"]
    imgs = [laser_img[:, :, ::-1],
            cv2.cvtColor(green_diff, cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(peak_mask,  cv2.COLOR_GRAY2RGB),
            overlay[:, :, ::-1]]
    for ax, img, t in zip(axes, imgs, titles):
        ax.imshow(img); ax.set_title(t, fontsize=9); ax.axis("off")

    plt.suptitle(f"Frame {frame_idx:04d}  |  {len(centroids)} laser rows detected", fontsize=11)
    plt.tight_layout()
    out = _subdir("laser") / f"frame{frame_idx:04d}.png"
    plt.savefig(out, dpi=130); plt.close()
    if centroids:
        rows = [r for _, r in centroids]
        print(f"Saved {out}  ({len(centroids)} rows, y={min(rows):.0f}–{max(rows):.0f})")
    else:
        print(f"Saved {out}  (0 rows detected)")


# ───────────────────────────────────────────────────────────────────────
# TOOL 2 — calibration overlay
# Projects the 6 hex-corner world points back onto Reference.png with
# both the manually-specified image points (red) and the solvePnP
# reprojections (green crosses), so you can see any mismatch.
# ───────────────────────────────────────────────────────────────────────
def tool_calib():
    R_cw, t_cw, laser_ab = calibrate(save_debug=False)
    ref = cv2.imread(str(DATA / "Reference.png"))

    s60 = 25 * np.sin(np.radians(60)); c60 = 25 * np.cos(np.radians(60))
    image_pts = np.array([[468.1,870.7],[645.5,911.2],[573.5,977.0],
                           [224.7,984.9],[571.3,1183.5],[223.6,1195.9]])
    world_pts = np.array([[c60,s60,0],[25,0,0],[c60,-s60,0],
                           [-c60,-s60,0],[c60,-s60,-15],[-c60,-s60,-15]])
    labels = ["60deg top","0deg top","300deg top","240deg top","300deg bot","240deg bot"]  # ASCII: cv2.putText cannot draw θ

    import cv2 as _cv2
    rvec, _ = _cv2.Rodrigues(R_cw)
    proj, _ = _cv2.projectPoints(world_pts, rvec, t_cw, K, None)
    proj = proj.reshape(-1, 2)

    dbg = ref.copy()
    for ip, lbl in zip(image_pts.astype(int), labels):
        _cv2.circle(dbg, tuple(ip), 8, (0, 0, 255), 2)
        _cv2.putText(dbg, lbl, (ip[0]+10, ip[1]-5),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    for pp, wp, lbl in zip(proj.astype(int), world_pts, labels):
        _cv2.drawMarker(dbg, tuple(pp), (0, 255, 0), _cv2.MARKER_CROSS, 14, 2)
        err = np.linalg.norm(pp - image_pts[labels.index(lbl)])
        _cv2.putText(dbg, f"{err:.1f}px", (pp[0]+10, pp[1]+15),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Also draw detected laser centroid line in Reference
    b, g, r = cv2.split(ref)
    gex = g.astype(np.int32) - np.maximum(r, b).astype(np.int32)
    lmask = (gex > 30) & (g > 80)
    for col in range(ref.shape[1]):
        rows = np.where(lmask[:, col])[0]
        if len(rows) >= 2:
            w = g[rows, col].astype(float)
            cr = int(np.average(rows, weights=w))
            _cv2.circle(dbg, (col, cr), 1, (0, 255, 255), -1)

    out = _subdir("calib") / "calibration_annotated.png"
    cv2.imwrite(str(out), dbg)
    print(f"Saved {out}")
    print("RED circles = manually specified image corners")
    print("GREEN crosses = solvePnP reprojections")
    print("YELLOW dots = detected laser line on Reference")


# ───────────────────────────────────────────────────────────────────────
# TOOL 3 — single-frame 3D slice
# Triangulates one frame and plots the resulting 3D points so you can
# inspect the shape of a single laser slice through the gnome.
# ───────────────────────────────────────────────────────────────────────
def tool_slice(frame_idx=0, peak_fraction=0.5, min_peak=20):
    R_cw, t_cw, laser_ab = calibrate(save_debug=False)
    cam_origin = -R_cw.T @ t_cw

    laser_img = cv2.imread(str(DATA / f"{frame_idx:04d}.jpg"))
    color_img = cv2.imread(str(DATA / "withoutLaser" / f"{frame_idx:04d}.jpg"))
    centroids = detect_laser(laser_img, color_img,
                             peak_fraction=peak_fraction, min_peak=min_peak)

    pts3d, colors = [], []
    for col, row in centroids:
        pt = triangulate(col, row, cam_origin, R_cw, laser_ab)
        if pt is None: continue
        if pt[2] < -16 or pt[2] > 85 or np.hypot(pt[0], pt[1]) > 28: continue
        pts3d.append(pt)
        u = int(np.clip(col, 0, color_img.shape[1]-1))
        v = int(np.clip(round(row), 0, color_img.shape[0]-1))
        bgr = color_img[v, u]
        colors.append((bgr[2]/255, bgr[1]/255, bgr[0]/255))

    pts3d = np.array(pts3d); colors = np.array(colors)
    print(f"Frame {frame_idx:04d}: {len(pts3d)} 3D points, "
          f"Z={pts3d[:,2].min():.1f}–{pts3d[:,2].max():.1f} mm")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (xi, yi, xl, yl) in zip(axes, [(0,2,"X","Z"),(1,2,"Y","Z"),(0,1,"X","Y")]):
        ax.scatter(pts3d[:,xi], pts3d[:,yi], c=colors, s=4)
        ax.set_xlabel(xl+" (mm)"); ax.set_ylabel(yl+" (mm)")
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.axvline(0, color="gray", lw=0.5, ls="--")
    plt.suptitle(f"Frame {frame_idx:04d} — single laser slice  (θ={frame_idx*2}°)", fontsize=11)
    plt.tight_layout()
    out = _subdir("slice") / f"frame{frame_idx:04d}.png"
    plt.savefig(out, dpi=130); plt.close()
    print(f"Saved {out}")


# ───────────────────────────────────────────────────────────────────────
# TOOL 4 — accumulated cloud check
# Loads the existing output.xyz and shows four views plus Z histogram
# so you can quickly spot bad clusters or wrong Z ranges.
# ───────────────────────────────────────────────────────────────────────
def tool_cloud(xyz_file="output.xyz"):
    pts = np.loadtxt(xyz_file)
    x,y,z = pts[:,0], pts[:,1], pts[:,2]
    col = np.column_stack([pts[:,3]/255, pts[:,4]/255, pts[:,5]/255])

    fig = plt.figure(figsize=(18, 10))

    # 2-D projections
    for i, (xi, yi, xl, yl) in enumerate([(0,2,"X","Z"),(1,2,"Y","Z"),(0,1,"X","Y")]):
        ax = fig.add_subplot(2, 3, i+1)
        ax.scatter(pts[:,xi][::4], pts[:,yi][::4], c=col[::4], s=0.3)
        ax.set_xlabel(xl+" (mm)"); ax.set_ylabel(yl+" (mm)")
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

    # Z histogram
    ax = fig.add_subplot(2, 3, 4)
    ax.hist(z, bins=60, color="steelblue", edgecolor="none")
    ax.axvline(0,  color="red",   lw=1, label="turntable top (Z=0)")
    ax.axvline(-15, color="orange", lw=1, label="turntable bot (Z=−15)")
    ax.set_xlabel("Z (mm)"); ax.set_ylabel("point count")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # r_xy histogram
    rxy = np.hypot(x, y)
    ax = fig.add_subplot(2, 3, 5)
    ax.hist(rxy, bins=60, color="tomato", edgecolor="none")
    ax.axvline(25, color="blue", lw=1, label="turntable radius (25 mm)")
    ax.set_xlabel("r_xy (mm)"); ax.set_ylabel("point count")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # stats text
    ax = fig.add_subplot(2, 3, 6)
    ax.axis("off")
    stats = (f"File: {xyz_file}\n"
             f"Total points: {len(pts):,}\n\n"
             f"X:   {x.min():.1f} – {x.max():.1f} mm\n"
             f"Y:   {y.min():.1f} – {y.max():.1f} mm\n"
             f"Z:   {z.min():.1f} – {z.max():.1f} mm\n"
             f"r_xy: {rxy.min():.1f} – {rxy.max():.1f} mm\n\n"
             f"Z<-15: {(z<-15).sum():,} pts (below turntable)\n"
             f"Z>75:  {(z>75).sum():,} pts (above gnome)\n"
             f"r>25:  {(rxy>25).sum():,} pts (outside turntable)")
    ax.text(0.05, 0.95, stats, transform=ax.transAxes,
            fontsize=10, va="top", family="monospace")

    plt.suptitle("Point cloud overview — output.xyz", fontsize=12)
    plt.tight_layout()
    out = _subdir("cloud") / "cloud.png"
    plt.savefig(out, dpi=130); plt.close()
    print(f"Saved {out}")


# ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tool", choices=["laser","calib","slice","cloud"])
    p.add_argument("--frame",        type=int,   default=0)
    p.add_argument("--peak_fraction",type=float, default=0.5,
                   help="keep pixels >= peak * fraction (default 0.5)")
    p.add_argument("--min_peak",     type=int,   default=20,
                   help="ignore columns where peak diff < this (default 20)")
    args = p.parse_args()

    import os; os.chdir(Path(__file__).parent)

    if args.tool == "laser":
        tool_laser(args.frame, args.peak_fraction, args.min_peak)
    elif args.tool == "calib":
        tool_calib()
    elif args.tool == "slice":
        tool_slice(args.frame, args.peak_fraction, args.min_peak)
    elif args.tool == "cloud":
        tool_cloud()
