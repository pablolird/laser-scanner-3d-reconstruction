# Development Log — 3D Laser Triangulation Reconstruction

**Goal:** Reconstruct a colored 3D point cloud of a gnome figurine from 180 laser-scanned frames.

---

## Overview of the Pipeline

The reconstruction has five phases:

1. **Calibration** — Estimate camera extrinsics (R, t) via solvePnP on hex turntable corners, then fit the laser plane equation `ax + by = 0` from the laser line visible on the turntable top face in `Reference.png`.
2. **Laser detection** — Per frame, subtract the no-laser image from the laser image (green channel), detect the laser line centroid per column.
3. **Triangulation** — For each detected laser pixel `(u, v)`, unproject a ray and intersect it with the fixed laser plane to get a 3D world point.
4. **Unrotation** — Rotate each world point by `−θᵢ` (the frame's turntable angle) to map it into the object's rest frame.
5. **Output** — Write all colored 3D points to `output.xyz` (`X Y Z R G B`).

---

## Calibration

### Camera intrinsics
Given as:
```
K = [[720,   0, 360],
     [  0, 720, 640],
     [  0,   0,   1]]
```
Image size: 720 × 1280 px.

### Coordinate system convention
- **Z = 0** at the turntable top face (matching `ground_truth.ply` convention).
- **Z = −15 mm** at the turntable bottom face.
- Gnome sits on the top face and extends upward.

### Camera extrinsics — solvePnP
Six hexagon corner correspondences were identified manually from Harris corner detection on `Reference.png`:

| Image point (px) | World point (mm) | Corner |
|---|---|---|
| (468, 871) | (12.5, 21.65, 0) | θ=60° top |
| (645, 911) | (25.0, 0.0, 0) | θ=0° top |
| (573, 977) | (12.5, −21.65, 0) | θ=300° top |
| (224, 985) | (−12.5, −21.65, 0) | θ=240° top |
| (571, 1184) | (12.5, −21.65, −15) | θ=300° bottom |
| (223, 1196) | (−12.5, −21.65, −15) | θ=240° bottom |

**Result:** PnP reprojection error = **9.92 px**. Camera center at (−2.2, −77.9, 17.7) mm.

The reprojection error is higher than ideal (~2–3 px would be good). The main cause is that Harris corner detection gives approximate positions, and some corners — particularly the two extreme side corners of the hex — were difficult to match reliably.

### Laser plane calibration
The laser line on the turntable top face (`Reference.png`) was unprojected to Z = 0, giving 241 3D points. SVD fit yielded:

```
Laser plane:  0.8797·x − 0.4755·y = 0
```

The laser plane direction in XY is at ~61° from the X-axis, consistent with the laser exiting the top face at the θ=60° hex corner.

---

## Laser Detection — Evolution

### v1: Fixed green threshold
```python
mask = green_diff > threshold   # threshold = 25, later 8
centroid = weighted_average(rows where mask is True)
```

**Problems found via debug tools:**
- At threshold=8: diffuse low-intensity blobs from turntable edge reflections were included. Per-column centroid was pulled away from the true laser center.
- At threshold=25: too aggressive — missed the laser on dark-colored gnome surfaces (e.g. black boots, back side).
- Neither value was stable across different surfaces and lighting conditions.

### v2: Peak-based detection with argmax (per column)
```python
peak = max(green_diff[:, col])
if peak < min_peak: skip column
centroid = argmax(green_diff[:, col])   # single brightest row
```

**Parameters:** `min_peak=25`

**Improvement over v1:** Replaced weighted centroid with argmax. The centroid was being pulled by stray noise pixels on dim columns — even with peak masking, when peak ≈ 10–15 DN the noise pixels at 40% cutoff (4–6 DN) are indistinguishable from signal. Argmax always picks the single brightest row, making it immune to scattered low-value noise.

**Remaining limitation:** Per-column scanning gives at most 720 points per frame. Dark surfaces cause the peak to fall below `min_peak=25` and whole columns drop out.

### v3: Argmax per row — current
```python
peak = max(green_diff[row, :])
if peak < min_peak: skip row
detection = (argmax(green_diff[row, :]), row)
```

**Key insight:** The laser stripe is nearly **vertical** in the image. Scanning per row rather than per column is the natural fit:
- Each row has one well-defined horizontal laser position.
- The image is 1280 px tall vs 720 px wide → **1280 samples per frame** instead of 720 (1280 × 180 = 230,400 theoretical max vs 129,600).
- The stripe can curve past a column entirely on complex geometry (gnome face), but it always crosses every row it illuminates.

**Result:** Point count went from ~44k to ~150k, with much denser coverage of the figurine. This was the single most impactful change in the pipeline. (Accuracy is limited by calibration, not detection. See Results.)

---

## 3D Filters — Evolution

After triangulation each world point `(x, y, z)` is checked against:

| Parameter | v1 | v2 (current) | Reason |
|---|---|---|---|
| Z_MIN | −5 mm | −16 mm | Allow turntable side face (bottom at Z=−15) |
| Z_MAX | 200 mm | 85 mm | Remove high-Z outliers above gnome hat |
| R_XY_MAX | 80 mm | 28 mm | Gnome cannot extend beyond turntable radius (25 mm) |

**Key finding from `debug_tools.py cloud`:** With R_XY_MAX=60 mm, 12,378 points were outside the physical turntable radius. These were all false detections from specular reflections off the turntable edges. Tightening to 28 mm eliminated them completely.

---

## Results

| Version | Points | Median C2C vs GT | Notes |
|---|---|---|---|
| v1 (threshold=25) | 22,170 | 5.7 mm | Too few points on dark surfaces |
| v1 (threshold=8) | 52,565 | 6.3 mm | Many outliers from turntable edge blobs |
| v2 peak-based per-column | 44,343 | — | Cleaner detection, capped at 720 pts/frame |
| **v3 argmax per-row** | **156,753** | **5.5 mm** (point-to-surface: 3.2 mm) | Per-row unlocks 1280 samples/frame; far denser coverage |

The v1 figures and the 5.5 mm v3 figure are cloud-to-cloud distances (nearest ground-truth *vertex*). `evaluate.py` measures distance to the nearest ground-truth *surface* (like CloudCompare's cloud-to-mesh): mean 4.73 mm, median 3.24 mm, 95th percentile 16.6 mm. The remaining error is mostly a vertical stretch (the cloud reaches Z = 85 mm vs 69 mm in the ground truth), which points to the ~9.8 px calibration error rather than laser detection.

**Remaining issue:** Some surface regions are not reconstructed — likely occluded areas never visible to the camera while the laser also illuminates them, or surfaces where the green reflection falls below `min_peak=25`.

---

## Debug Tools (`debug_tools.py`)

Four tools built to supervise each step. All outputs go into per-tool subfolders:

```
debug/
├── laser/   frame0000.png, frame0045.png, ...
├── calib/   calibration.png, calibration_annotated.png
├── slice/   frame0000.png, frame0045.png, ...
└── cloud/   cloud.png
```

```bash
# Inspect laser detection quality on any frame
python debug_tools.py laser --frame N --min_peak 25

# Check calibration: red=manual corners, green=reprojections, yellow=laser
python debug_tools.py calib

# See one triangulated laser slice in 3D
python debug_tools.py slice --frame N --min_peak 25

# Full cloud statistics: Z histogram, r_xy histogram, 2D projections
python debug_tools.py cloud
```

The `laser` tool shows: raw frame / green diff / peak mask / detected points. Detected points are drawn as **red dots (radius 3)** with a **transparent yellow row highlight** so it's easy to verify coverage against the visible laser stripe.

---

## Known Issues / Open Questions

1. **Missing regions:** Some surface patches are not reconstructed. Two possible causes: (a) occlusion — the camera never sees that surface while the laser hits it, addressed partially by the full 360° rotation; (b) dark surfaces with green reflection below `min_peak=25` — lowering the threshold recovers them but risks adding noise.

2. **Calibration error (9.92 px):** Ideally < 3 px. Would require more precise corner identification, possibly using edge-line fitting on the hex faces instead of Harris corners.

3. **Turntable surface points included:** Points on the turntable side faces (Z < 0) are included in the output. Physically correct but add noise to the C2C comparison against the gnome-only ground truth.
