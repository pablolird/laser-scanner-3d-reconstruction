#!/usr/bin/env python3
"""
pick_calibration_pts.py
Interactively pick the 6 hex-turntable corners in Reference.png,
then compute solvePnP and print the updated image_pts for reconstruct.py.

Pick order (matches world_pts in reconstruct.py):
  1. θ= 60° top-face corner  → world ( 12.5,  21.65,   0)
  2. θ=  0° top-face corner  → world ( 25.0,   0.0,    0)
  3. θ=300° top-face corner  → world ( 12.5, -21.65,   0)
  4. θ=240° top-face corner  → world (-12.5, -21.65,   0)
  5. θ=300° bottom-face corner → world ( 12.5, -21.65, -15)
  6. θ=240° bottom-face corner → world (-12.5, -21.65, -15)

Controls
--------
  Click               Jump crosshair to that position
  Arrow / Shift+Arrow Move 1 px / 10 px
  Enter               Confirm current point
  U                   Undo last point
  W A S D             Pan view
  + - / scroll        Zoom in / out
  C                   Centre view on crosshair
  V                   Toggle Fine/Fast speed
  Q / Esc             Finish (also works after all 6 are picked)
"""

import os, sys
import tkinter as tk
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageTk

# ─── constants (same as reconstruct.py) ──────────────────────────────────
CANVAS_W      = 1000
CANVAS_H      = 700
ZOOM_STEP     = 1.25
ZOOM_MIN      = 0.05
ZOOM_MAX      = 32.0
PAN_STEP      = 30
CROSSHAIR_ARM = 22
POINT_RADIUS  = 7

DARK_BG  = '#1a1a1a'
PANEL_BG = '#242424'
BAR_BG   = '#2d2d2d'
ACTIVE_BD= '#00aaff'

COLORS = [
    '#FF4444','#44DD44','#4488FF','#FFDD00','#FF44FF',
    '#44FFFF','#FF8800','#88FF44','#AA44FF','#FF4488',
]

# ─── corner definitions ───────────────────────────────────────────────────────
HEX_R = 25.0
s60   = HEX_R * np.sin(np.radians(60))   # 21.651 mm
c60   = HEX_R * np.cos(np.radians(60))   # 12.5   mm

CORNER_INFO = [
    ("θ=60°  top",    ( c60,  s60,   0.0), (468, 871)),
    ("θ=0°   top",    (25.0,  0.0,   0.0), (645, 911)),
    ("θ=300° top",    ( c60, -s60,   0.0), (573, 977)),
    ("θ=240° top",    (-c60, -s60,   0.0), (224, 985)),
    ("θ=300° bottom", ( c60, -s60, -15.0), (571, 1184)),
    ("θ=240° bottom", (-c60, -s60, -15.0), (223, 1196)),
]
N_CORNERS = len(CORNER_INFO)

K = np.array([[720., 0., 360.],
              [  0., 720., 640.],
              [  0.,   0.,   1.]], dtype=np.float64)



# ─── ImagePanel ───────────────────────────────────────────────────────────────
class ImagePanel:
    def __init__(self, parent, image_path):
        self.image_path = image_path
        self.image_pil  = Image.open(image_path).convert('RGB')
        self.img_w, self.img_h = self.image_pil.size
        self.zoom  = self._fit_zoom()
        self.pan_x = self.img_w / 2.0
        self.pan_y = self.img_h / 2.0
        self.cur_x = float(self.img_w // 2)
        self.cur_y = float(self.img_h // 2)
        self.selected_points: list[tuple[int, int]] = []
        self._photo_ref = None

        self.outer = tk.Frame(parent, bd=3, bg=ACTIVE_BD)
        self.outer.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.canvas = tk.Canvas(
            self.outer, width=CANVAS_W, height=CANVAS_H,
            bg='#111111', cursor='crosshair', highlightthickness=0)
        self.canvas.pack()
        self.coord_var = tk.StringVar(value='x: —  y: —')
        tk.Label(self.outer, textvariable=self.coord_var,
                 bg=PANEL_BG, fg='#aaffaa', font=('Courier', 10),
                 anchor='w', padx=6, pady=3).pack(fill=tk.X)

    def _fit_zoom(self):
        return max(ZOOM_MIN, min((CANVAS_W-20)/self.img_w, (CANVAS_H-20)/self.img_h, 1.0))

    def img_to_canvas(self, ix, iy):
        return (CANVAS_W/2 + (ix - self.pan_x)*self.zoom,
                CANVAS_H/2 + (iy - self.pan_y)*self.zoom)

    def canvas_to_img(self, cx, cy):
        return (self.pan_x + (cx - CANVAS_W/2)/self.zoom,
                self.pan_y + (cy - CANVAS_H/2)/self.zoom)

    def clamp_cursor(self):
        self.cur_x = max(0.0, min(self.img_w - 1, self.cur_x))
        self.cur_y = max(0.0, min(self.img_h - 1, self.cur_y))

    def clamp_pan(self):
        pad = 50
        self.pan_x = max(-pad/self.zoom, min(self.img_w + pad/self.zoom, self.pan_x))
        self.pan_y = max(-pad/self.zoom, min(self.img_h + pad/self.zoom, self.pan_y))

    def autopan_to_cursor(self):
        margin = max(40, CROSSHAIR_ARM + 10)
        cx, cy = self.img_to_canvas(self.cur_x, self.cur_y)
        if cx < margin:              self.pan_x -= (margin - cx) / self.zoom
        elif cx > CANVAS_W - margin: self.pan_x += (cx - (CANVAS_W - margin)) / self.zoom
        if cy < margin:              self.pan_y -= (margin - cy) / self.zoom
        elif cy > CANVAS_H - margin: self.pan_y += (cy - (CANVAS_H - margin)) / self.zoom
        self.clamp_pan()

    def render(self, hint=''):
        zoom = self.zoom
        hw = CANVAS_W / (2.0 * zoom); hh = CANVAS_H / (2.0 * zoom)
        left = self.pan_x - hw; top = self.pan_y - hh
        right = self.pan_x + hw; bot = self.pan_y + hh
        cl = max(0.0, left); ct = max(0.0, top)
        cr = min(float(self.img_w), right); cb = min(float(self.img_h), bot)

        bg = Image.new('RGB', (CANVAS_W, CANVAS_H), (17, 17, 17))
        if cr > cl and cb > ct:
            cropped = self.image_pil.crop((int(cl), int(ct), int(cr), int(cb)))
            tw = max(1, int((cr - cl) * zoom)); th = max(1, int((cb - ct) * zoom))
            resample = Image.NEAREST if zoom >= 3 else Image.LANCZOS
            bg.paste(cropped.resize((tw, th), resample),
                     (int((cl - left)*zoom), int((ct - top)*zoom)))

        self._photo_ref = ImageTk.PhotoImage(bg)
        c = self.canvas
        c.delete('all')
        c.create_image(0, 0, anchor=tk.NW, image=self._photo_ref)

        for i, (px, py) in enumerate(self.selected_points):
            cx, cy = self.img_to_canvas(px, py)
            col = COLORS[i % len(COLORS)]
            c.create_oval(cx - POINT_RADIUS, cy - POINT_RADIUS,
                          cx + POINT_RADIUS, cy + POINT_RADIUS,
                          fill=col, outline='white', width=2)
            c.create_text(cx + POINT_RADIUS + 4, cy, text=str(i + 1),
                          fill=col, font=('Arial', 10, 'bold'), anchor=tk.W)

        cxh, cyh = self.img_to_canvas(self.cur_x, self.cur_y)
        col, gap, arm = '#00ff88', 5, CROSSHAIR_ARM
        c.create_line(cxh - arm, cyh, cxh - gap, cyh, fill=col, width=1)
        c.create_line(cxh + gap, cyh, cxh + arm, cyh, fill=col, width=1)
        c.create_line(cxh, cyh - arm, cxh, cyh - gap, fill=col, width=1)
        c.create_line(cxh, cyh + gap, cxh, cyh + arm, fill=col, width=1)
        c.create_oval(cxh - 2, cyh - 2, cxh + 2, cyh + 2, fill=col, outline=col)

        ix, iy = int(round(self.cur_x)), int(round(self.cur_y))
        self.coord_var.set(
            f'x (col): {ix:4d}   y (row): {iy:4d}   |   zoom: {zoom:.2f}×'
            + (f'   |   {hint}' if hint else ''))


# ─── CalibPicker app ──────────────────────────────────────────────────────────
class CalibPicker:
    def __init__(self, root, guide_path):
        self.root   = root
        self.pts    = []   # collected (x, y) floats
        self.speed  = 1

        self.root.configure(bg=DARK_BG)
        self.root.resizable(False, False)
        self.root.title("Calibration corner picker")

        self.title_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.title_var,
                 bg=BAR_BG, fg='white', font=('Arial', 11),
                 anchor='w', padx=10, pady=6).pack(fill=tk.X)

        frame = tk.Frame(self.root, bg=DARK_BG)
        frame.pack(fill=tk.BOTH, expand=True)
        self.panel = ImagePanel(frame, guide_path)

        hints = tk.Frame(self.root, bg=BAR_BG)
        hints.pack(fill=tk.X)
        tk.Label(hints,
                 text='Click: jump   Arrows/Shift: move 1/10px   Enter: confirm   '
                      'U: undo   WASD: pan   +/-/scroll: zoom   C: centre   V: speed   Q: done',
                 bg=BAR_BG, fg='#888888', font=('Arial', 9),
                 padx=10, pady=4).pack(side=tk.LEFT)

        self.speed_var = tk.StringVar(value='Speed: FINE (1px)')
        tk.Button(hints, textvariable=self.speed_var,
                  bg='#3a3a3a', fg='#aaffaa', font=('Arial', 9, 'bold'),
                  relief=tk.FLAT, padx=8, pady=2, cursor='hand2',
                  command=self._toggle_speed).pack(side=tk.RIGHT, padx=8, pady=3)

        self.status_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.status_var,
                 bg='#111111', fg='#aaffaa', font=('Courier', 10),
                 anchor='w', padx=8, pady=3, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, side=tk.BOTTOM)

        b = self.root.bind
        b('<Left>',        lambda e: self._move(-1, 0))
        b('<Right>',       lambda e: self._move(1,  0))
        b('<Up>',          lambda e: self._move(0, -1))
        b('<Down>',        lambda e: self._move(0,  1))
        b('<Shift-Left>',  lambda e: self._move(-10, 0))
        b('<Shift-Right>', lambda e: self._move(10,  0))
        b('<Shift-Up>',    lambda e: self._move(0, -10))
        b('<Shift-Down>',  lambda e: self._move(0,  10))
        b('<Return>',      lambda e: self._confirm())
        b('<KP_Enter>',    lambda e: self._confirm())
        b('<u>',           lambda e: self._undo())
        b('<plus>',        lambda e: self._zoom(ZOOM_STEP))
        b('<equal>',       lambda e: self._zoom(ZOOM_STEP))
        b('<minus>',       lambda e: self._zoom(1/ZOOM_STEP))
        b('<w>',           lambda e: self._pan(0, -PAN_STEP))
        b('<s>',           lambda e: self._pan(0,  PAN_STEP))
        b('<a>',           lambda e: self._pan(-PAN_STEP, 0))
        b('<d>',           lambda e: self._pan( PAN_STEP, 0))
        b('<c>',           lambda e: self._centre())
        b('<v>',           lambda e: self._toggle_speed())
        b('<q>',           lambda e: self._quit())
        b('<Escape>',      lambda e: self._quit())
        self.panel.canvas.bind('<Button-1>',   self._on_click)
        self.panel.canvas.bind('<MouseWheel>', self._on_scroll)
        self.panel.canvas.bind('<Button-4>',   lambda e: self._zoom(ZOOM_STEP))
        self.panel.canvas.bind('<Button-5>',   lambda e: self._zoom(1/ZOOM_STEP))

        # jump crosshair to first approximate location
        self.panel.cur_x = float(CORNER_INFO[0][2][0])
        self.panel.cur_y = float(CORNER_INFO[0][2][1])
        self.panel.pan_x = self.panel.cur_x
        self.panel.pan_y = self.panel.cur_y
        self._refresh()

    def _next_label(self):
        n = len(self.pts)
        if n >= N_CORNERS:
            return "All corners picked — press Q to finish"
        label, world, approx = CORNER_INFO[n]
        return (f"Point {n+1}/{N_CORNERS}: {label}   "
                f"world=({world[0]:.1f}, {world[1]:.1f}, {world[2]:.1f}) mm   "
                f"approx pixel≈({approx[0]}, {approx[1]})")

    def _refresh(self):
        n = len(self.pts)
        self.title_var.set(f"{n}/{N_CORNERS} corners picked")
        hint = '' if n >= N_CORNERS else CORNER_INFO[n][0]
        self.panel.render(hint=hint)
        self.status_var.set(self._next_label())

    def _on_click(self, event):
        p = self.panel
        ix, iy = p.canvas_to_img(event.x, event.y)
        p.cur_x = ix; p.cur_y = iy
        p.clamp_cursor()
        self._refresh()

    def _move(self, dx, dy):
        p = self.panel
        p.cur_x += dx * self.speed
        p.cur_y += dy * self.speed
        p.clamp_cursor()
        p.autopan_to_cursor()
        self._refresh()

    def _pan(self, dcx, dcy):
        p = self.panel
        p.pan_x += dcx / p.zoom
        p.pan_y += dcy / p.zoom
        p.clamp_pan()
        self._refresh()

    def _zoom(self, f):
        p = self.panel
        p.zoom = max(ZOOM_MIN, min(ZOOM_MAX, p.zoom * f))
        self._refresh()

    def _on_scroll(self, e):
        self._zoom(ZOOM_STEP if e.delta > 0 else 1/ZOOM_STEP)

    def _centre(self):
        p = self.panel
        p.pan_x = p.cur_x; p.pan_y = p.cur_y
        self._refresh()

    def _toggle_speed(self):
        self.speed = 10 if self.speed == 1 else 1
        self.speed_var.set('Speed: FAST (10px)' if self.speed == 10 else 'Speed: FINE (1px)')

    def _confirm(self):
        if len(self.pts) >= N_CORNERS:
            self.status_var.set(f"Already have all {N_CORNERS} points. Press Q to finish.")
            return
        p = self.panel
        x, y = p.cur_x, p.cur_y   # keep float for sub-pixel accuracy
        self.pts.append((x, y))
        p.selected_points.append((int(round(x)), int(round(y))))
        label = CORNER_INFO[len(self.pts) - 1][0]
        print(f"  Point {len(self.pts)}: {label} → ({x:.1f}, {y:.1f})")
        # jump crosshair to next approximate location
        if len(self.pts) < N_CORNERS:
            nx, ny = CORNER_INFO[len(self.pts)][2]
            p.cur_x = float(nx); p.cur_y = float(ny)
            p.pan_x = p.cur_x;   p.pan_y = p.cur_y
        self._refresh()

    def _undo(self):
        if self.pts:
            self.pts.pop()
            self.panel.selected_points.pop()
            # jump back to approximate location of the undone point
            nx, ny = CORNER_INFO[len(self.pts)][2]
            self.panel.cur_x = float(nx); self.panel.cur_y = float(ny)
            self.panel.pan_x = self.panel.cur_x; self.panel.pan_y = self.panel.cur_y
            print(f"  Undo — back to point {len(self.pts)+1}")
            self._refresh()

    def _quit(self):
        self.root.destroy()


# ─── run solvePnP and update reconstruct.py ───────────────────────────────────
def evaluate_and_patch(image_pts: np.ndarray):
    world_pts = np.array([info[1] for info in CORNER_INFO], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(world_pts, image_pts, K, None,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        print("\nsolvePnP FAILED — check your point correspondences.")
        return

    proj, _ = cv2.projectPoints(world_pts, rvec, tvec, K, None)
    err = float(np.linalg.norm(proj.reshape(-1, 2) - image_pts, axis=1).mean())
    R, _ = cv2.Rodrigues(rvec)
    cam  = -R.T @ tvec.ravel()

    print(f"\n{'─'*60}")
    print(f"  solvePnP reprojection error : {err:.2f} px  (was 9.92 px)")
    print(f"  Camera centre (world)       : {cam.round(2)} mm")
    print(f"{'─'*60}\n")

    # format the new image_pts block
    lines = ["    image_pts = np.array([\n"]
    for i, (x, y) in enumerate(image_pts):
        label, world, _ = CORNER_INFO[i]
        lines.append(f"        [{x:.1f}, {y:.1f}],   # {label} → {world}\n")
    lines.append("    ], dtype=np.float64)\n")
    new_block = "".join(lines)

    print("New image_pts block to paste into reconstruct.py:\n")
    print(new_block)

    # patch reconstruct.py automatically
    rpath = Path(__file__).parent / "reconstruct.py"
    src   = rpath.read_text()

    old_start = "    image_pts = np.array(["
    old_end   = "    ], dtype=np.float64)\n"
    s = src.find(old_start)
    e = src.find(old_end, s) + len(old_end)
    if s == -1 or e == -1:
        print("Could not locate image_pts block in reconstruct.py — paste manually.")
        return

    new_src = src[:s] + new_block + src[e:]
    rpath.write_text(new_src)
    print(f"reconstruct.py updated automatically. Run it to rebuild the cloud.")


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    os.chdir(Path(__file__).parent)
    ref_path = str(Path("data") / "Reference.png")

    print("Opening picker — zoom in (+/scroll), fine-tune with arrow keys, confirm with Enter.\n")
    for i, (label, world, approx) in enumerate(CORNER_INFO):
        print(f"  {i+1}. {label:20s}  world={world}  approx pixel≈{approx}")
    print()

    root = tk.Tk()
    app  = CalibPicker(root, ref_path)
    root.mainloop()

    pts = app.pts
    if len(pts) < N_CORNERS:
        print(f"\nOnly {len(pts)}/{N_CORNERS} points picked — exiting without changes.")
        return

    image_pts = np.array(pts, dtype=np.float64)
    evaluate_and_patch(image_pts)


if __name__ == "__main__":
    main()
