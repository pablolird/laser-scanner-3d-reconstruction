#!/usr/bin/env python3
"""
Render the README animations into docs/:
  docs/scan.gif       input laser frames next to the point cloud as it builds up
  docs/turntable.gif  the finished colored point cloud spinning 360°

Usage:  python make_gifs.py
"""

import os
import cv2
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

import reconstruct as rc

DOCS = Path("docs")
BG   = "#111111"
RNG  = np.random.default_rng(0)


def frame_points(i, cam_origin, R_cw, laser_ab):
    """Run detection + triangulation for one frame; returns (N,6) XYZRGB in the object frame."""
    laser_img = cv2.imread(str(rc.DATA / f"{i:04d}.jpg"))
    color_img = cv2.imread(str(rc.DATA / "withoutLaser" / f"{i:04d}.jpg"))
    R_inv = rc.rotation_z(-i * 2.0)
    out = []
    for col, row in rc.detect_laser(laser_img, color_img):
        pt = rc.triangulate(col, row, cam_origin, R_cw, laser_ab)
        if pt is None or not (-16.0 <= pt[2] <= 85.0) or np.hypot(pt[0], pt[1]) > 28.0:
            continue
        b, g, r = color_img[int(row), int(round(col))]
        out.append((*(R_inv @ pt), r, g, b))
    return np.array(out).reshape(-1, 6), laser_img


def styled_axes(ax):
    """Dark, axis-free 3D view with equal aspect around the turntable."""
    ax.set_facecolor(BG)
    ax.set_axis_off()
    ax.set_xlim(-28, 28); ax.set_ylim(-28, 28); ax.set_zlim(-16, 86)
    ax.set_box_aspect((56, 56, 102))


def colors(pts):
    # the no-laser renders are dim; brighten for display only
    return np.clip(pts[:, 3:6] / 255.0 * 1.6, 0, 1)


def make_scan_gif(per_frame, laser_imgs, step=3):
    """Left: raw laser frame. Right: cloud accumulated so far."""
    fig = plt.figure(figsize=(7, 5.2), facecolor=BG)
    ax_img = fig.add_axes([0.0, 0.0, 0.42, 1.0]); ax_img.set_axis_off()
    ax3d = fig.add_axes([0.40, 0.0, 0.60, 1.0], projection="3d")
    idx = list(range(0, rc.N_FRAMES, step))

    def draw(k):
        i = idx[k]
        ax_img.clear(); ax_img.set_axis_off()
        img = cv2.cvtColor(laser_imgs[i][150:1250], cv2.COLOR_BGR2RGB)
        ax_img.imshow(img)
        ax_img.set_title(f"frame {i:03d}  ·  {i*2}°", color="w", fontsize=10)
        ax3d.clear(); styled_axes(ax3d)
        pts = np.concatenate(per_frame[: i + 1])
        if len(pts) > 30000:
            pts = pts[RNG.choice(len(pts), 30000, replace=False)]
        ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors(pts), s=0.4, linewidths=0)
        ax3d.view_init(elev=15, azim=-60)
        ax3d.set_title(f"{sum(len(p) for p in per_frame[: i + 1]):,} points",
                       color="w", fontsize=10, y=0.95)

    FuncAnimation(fig, draw, frames=len(idx)).save(
        DOCS / "scan.gif", writer=PillowWriter(fps=12), dpi=80)
    plt.close(fig)


def make_turntable_gif(cloud, n_views=72):
    """Spin the finished cloud 360° around Z."""
    if len(cloud) > 50000:
        cloud = cloud[RNG.choice(len(cloud), 50000, replace=False)]
    fig = plt.figure(figsize=(4.5, 6), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    styled_axes(ax)
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=colors(cloud), s=0.5, linewidths=0)

    def draw(k):
        ax.view_init(elev=12, azim=-90 + k * 360 / n_views)

    FuncAnimation(fig, draw, frames=n_views).save(
        DOCS / "turntable.gif", writer=PillowWriter(fps=15), dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    DOCS.mkdir(exist_ok=True)
    R_cw, t_cw, laser_ab = rc.calibrate(save_debug=False)
    cam_origin = -R_cw.T @ t_cw

    per_frame, laser_imgs = [], []
    for i in range(rc.N_FRAMES):
        pts, img = frame_points(i, cam_origin, R_cw, laser_ab)
        per_frame.append(pts); laser_imgs.append(img)

    print("Rendering docs/scan.gif ...");      make_scan_gif(per_frame, laser_imgs)
    print("Rendering docs/turntable.gif ..."); make_turntable_gif(np.concatenate(per_frame))
    print("Done.")
