#!/usr/bin/env python3
"""
3D Point Cloud Reconstruction via Laser Triangulation
Computer vision course project
"""

import cv2
import numpy as np
from pathlib import Path
import os

# ─────────────────────────── constants ───────────────────────────────

DATA   = Path("data")
OUTPUT = "output.xyz"

N_FRAMES = 180

K = np.array([[720.,  0., 360.],
              [  0., 720., 640.],
              [  0.,   0.,   1.]], dtype=np.float64)
K_INV = np.linalg.inv(K)

HEX_R     = 25.0   # mm  circumradius
HEX_Z_TOP =  0.0   # mm  top face  (origin = turntable top surface, matching GT)
HEX_Z_BOT = -15.0  # mm  bottom face

# ─────────────────────────── Phase 1: calibration ─────────────────────

def calibrate(save_debug=True):
    """
    Returns R_cw (3×3), t_cw (3,) and laser_ab (a,b) where ax+by=0 is
    the laser plane.
    """
    ref = cv2.imread(str(DATA / "Reference.png"))
    assert ref is not None, "Reference.png not found"

    # ── 1a. camera extrinsics via solvePnP ──────────────────────────
    #
    # Six hexagon corners picked by hand with pick_calibration_pts.py:
    # four top-face corners (Z=0) at θ=60°, 0°, 300°, 240° and two
    # bottom-face corners (Z=-15) at θ=300°, 240°.
    #
    s60 = HEX_R * np.sin(np.radians(60))   # 21.651 mm
    c60 = HEX_R * np.cos(np.radians(60))   # 12.5   mm

    image_pts = np.array([
        [471.0,  871.0],   # θ=60°  top    → ( 12.5,  21.65,   0)
        [648.0,  912.0],   # θ=0°   top    → ( 25.0,   0.0,    0)
        [573.0,  978.0],   # θ=300° top    → ( 12.5, -21.65,   0)
        [225.0,  986.0],   # θ=240° top    → (-12.5, -21.65,   0)
        [573.0, 1185.0],   # θ=300° bottom → ( 12.5, -21.65, -15)
        [224.0, 1197.0],   # θ=240° bottom → (-12.5, -21.65, -15)
    ], dtype=np.float64)

    world_pts = np.array([
        [ c60,   s60,  HEX_Z_TOP],
        [ HEX_R, 0.0,  HEX_Z_TOP],
        [ c60,  -s60,  HEX_Z_TOP],
        [-c60,  -s60,  HEX_Z_TOP],
        [ c60,  -s60,  HEX_Z_BOT],
        [-c60,  -s60,  HEX_Z_BOT],
    ], dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        world_pts, image_pts, K, None,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    assert ok, "solvePnP failed"

    R_cw, _ = cv2.Rodrigues(rvec)
    t_cw    = tvec.ravel()

    # reprojection error
    proj, _ = cv2.projectPoints(world_pts, rvec, tvec, K, None)
    err = float(np.linalg.norm(proj.reshape(-1,2) - image_pts, axis=1).mean())
    print(f"  PnP reprojection error: {err:.2f} px")

    cam_world = (-R_cw.T @ t_cw)
    print(f"  Camera center in world: {cam_world.round(1)} mm")

    # ── 1b. laser-plane calibration ─────────────────────────────────
    #
    # The laser is visible on the turntable TOP face (Z=0) in
    # Reference.png.  Unproject each centroid to Z=0 and fit ax+by=0.
    #
    b_ch, g_ch, r_ch = cv2.split(ref)
    green_excess = (g_ch.astype(np.int32)
                    - np.maximum(r_ch, b_ch).astype(np.int32))
    laser_mask = (green_excess > 30) & (g_ch > 80)

    top_xy = []
    for col in range(ref.shape[1]):  # 0..719
        rows = np.where(laser_mask[:, col])[0]
        if len(rows) < 2:
            continue
        w = g_ch[rows, col].astype(np.float64)
        row = float(np.average(rows, weights=w))

        pix_dir = K_INV @ np.array([col, row, 1.0])
        ray_dir = R_cw.T @ pix_dir            # in world coords
        dz = ray_dir[2]
        if abs(dz) < 1e-9:
            continue
        lam = (HEX_Z_TOP - cam_world[2]) / dz
        if lam < 0:
            continue
        pt = cam_world + lam * ray_dir

        # keep only points inside the turntable top face
        if np.hypot(pt[0], pt[1]) < HEX_R + 5.0:
            top_xy.append(pt[:2])

    top_xy = np.array(top_xy)
    print(f"  Laser top-face points for plane fit: {len(top_xy)}")

    # fit line through origin: smallest right-singular vector of top_xy
    _, _, Vt = np.linalg.svd(top_xy, full_matrices=False)
    laser_ab = Vt[-1]                        # (a, b): a*x + b*y = 0
    if laser_ab[0] < 0:
        laser_ab = -laser_ab                 # canonical sign

    print(f"  Laser plane: {laser_ab[0]:.4f}·x + {laser_ab[1]:.4f}·y = 0")

    # ── debug: save annotated reference image ───────────────────────
    if save_debug:
        dbg = ref.copy()
        for (ix, iy), (wx, wy, wz) in zip(image_pts.astype(int), world_pts):
            cv2.circle(dbg, tuple(ix_iy := (int(ix), int(iy))), 8, (0,0,255), 2)
            cv2.putText(dbg, f"({wx:.0f},{wy:.0f},{wz:.0f})",
                        (int(ix)+10, int(iy)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0,255,255), 1)
        # reproject world points
        for p in proj.reshape(-1, 2):
            cv2.drawMarker(dbg, (int(p[0]), int(p[1])), (0,255,0),
                           cv2.MARKER_CROSS, 12, 2)
        calib_dir = Path("debug") / "calib"; calib_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(calib_dir / "calibration.png"), dbg)
        print("  Saved debug/calib/calibration.png")

    return R_cw, t_cw, laser_ab


# ─────────────────────────── Phase 2: laser detection ─────────────────

def detect_laser(laser_img, color_img, peak_fraction=0.4, min_peak=25):
    """
    Brightest-pixel laser line detection, scanning per row.

    The laser stripe is nearly vertical, so each row has one well-defined
    horizontal laser position. Scanning rows gives up to 1280 points per
    frame vs 720 for per-column.
    peak_fraction is accepted but unused (kept for API compatibility with debug_tools).
    Returns a list of (col, row) pairs.
    """
    diff = laser_img.astype(np.int32) - color_img.astype(np.int32)
    green_diff = np.clip(diff[:, :, 1], 0, 255)

    detections = []
    for row in range(laser_img.shape[0]):
        row_diff = green_diff[row, :]
        peak = int(row_diff.max())
        if peak < min_peak:
            continue
        detections.append((float(np.argmax(row_diff)), float(row)))
    return detections


# ─────────────────────────── Phase 3: triangulation ───────────────────

def rotation_z(deg):
    th = np.radians(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.],
                     [s,  c, 0.],
                     [0., 0., 1.]])


def triangulate(col, row, cam_origin, R_cw, laser_ab):
    """
    Unproject pixel (col,row) and intersect ray with laser plane ax+by=0.
    Returns 3-D world point or None.
    """
    pix_dir   = K_INV @ np.array([col, row, 1.0])
    ray_world = R_cw.T @ pix_dir

    a, b = laser_ab
    ox, oy = cam_origin[0], cam_origin[1]
    dx, dy = ray_world[0], ray_world[1]

    denom = a * dx + b * dy
    if abs(denom) < 1e-9:
        return None

    lam = -(a * ox + b * oy) / denom
    if lam < 0:
        return None

    return cam_origin + lam * ray_world


# ─────────────────────────── Phase 4: main loop ───────────────────────

def reconstruct():
    print("Phase 1: Calibration ...")
    R_cw, t_cw, laser_ab = calibrate()

    cam_origin = -R_cw.T @ t_cw

    # Z=0 is turntable top face; turntable bottom is Z=-15; gnome ~0 to 85mm
    Z_MIN, Z_MAX  = -16.0, 85.0
    R_XY_MAX      = 28.0   # gnome cannot extend beyond turntable radius (25mm)

    points = []
    print("\nPhase 2-4: Reconstruction loop ...")

    for i in range(N_FRAMES):
        laser_path = DATA / f"{i:04d}.jpg"
        color_path = DATA / "withoutLaser" / f"{i:04d}.jpg"

        laser_img = cv2.imread(str(laser_path))
        color_img = cv2.imread(str(color_path))
        if laser_img is None or color_img is None:
            continue

        centroids = detect_laser(laser_img, color_img)
        theta_i   = i * 2.0
        R_inv     = rotation_z(-theta_i)   # un-rotate from world to object frame

        for col, row in centroids:
            pt = triangulate(col, row, cam_origin, R_cw, laser_ab)
            if pt is None:
                continue

            z    = pt[2]
            r_xy = np.hypot(pt[0], pt[1])
            if z < Z_MIN or z > Z_MAX or r_xy > R_XY_MAX:
                continue

            pt_obj = R_inv @ pt             # map to object's rest frame

            # sample colour from the no-laser image
            u = int(np.clip(round(col), 0, color_img.shape[1] - 1))
            v = int(np.clip(round(row), 0, color_img.shape[0] - 1))
            bgr = color_img[v, u]
            points.append((pt_obj[0], pt_obj[1], pt_obj[2],
                           int(bgr[2]), int(bgr[1]), int(bgr[0])))  # RGB

        if (i + 1) % 30 == 0:
            print(f"  Frame {i+1:3d}/{N_FRAMES}  accumulated {len(points)} pts")

    return points


def write_xyz(points, path=OUTPUT):
    print(f"\nWriting {len(points)} points → {path}")
    with open(path, "w") as f:
        for x, y, z, r, g, b in points:
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")
    print("Done.")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    pts = reconstruct()
    write_xyz(pts)
