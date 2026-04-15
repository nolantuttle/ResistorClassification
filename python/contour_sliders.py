"""
Contour tuning script — GrabCut params hardcoded, post-processing tunable.

Controls
  [ / ]   — prev / next image
  R       — rerun with current slider values
  S       — print current values to stdout
  Q       — quit
"""

import cv2 as cv
import numpy as np
import os
import sys

IMAGE_DIR = sys.argv[1] if len(sys.argv) > 1 else 'archive'
WIN       = 'Tune Contours'

# ── Hardcoded GrabCut params (tune in contour.py if needed) ──────────────────
GC_RECT_MARGIN_X = 0.40
GC_RECT_MARGIN_Y = 0.40
GC_ITER          = 7


def nothing(x): pass


all_images = []
for root, dirs, files in os.walk(IMAGE_DIR):
    for f in files:
        if f.endswith('.jpg'):
            all_images.append(os.path.join(root, f))
all_images.sort()

assert len(all_images) > 0, f"No jpg images found in {IMAGE_DIR}"
print(f"Found {len(all_images)} images.")
print("R=rerun  [/]=prev/next  S=print values  Q=quit")

img_index = [0]


def load_img():
    return cv.imread(all_images[img_index[0]])


cv.namedWindow(WIN, cv.WINDOW_NORMAL)
cv.resizeWindow(WIN, 1400, 800)

# Numbered so they're identifiable even if Qt drops label text
cv.createTrackbar('1 Close size',    WIN,  5, 30, nothing)
cv.createTrackbar('2 Close iter',    WIN,  3,  8, nothing)
cv.createTrackbar('3 Open size',     WIN,  4, 30, nothing)
cv.createTrackbar('4 Open iter',     WIN,  2,  8, nothing)
cv.createTrackbar('5 Min area',      WIN, 677, 9000, nothing)
cv.createTrackbar('6 Center zone%',  WIN, 31, 100, nothing)
cv.createTrackbar('7 Erode size',    WIN,  8, 40, nothing)
cv.createTrackbar('8 Erode iter',    WIN,  1, 10, nothing)
cv.createTrackbar('9 Aspect min*10', WIN, 11, 50, nothing)
cv.createTrackbar('10 Aspect max*10',WIN, 80, 150, nothing)
cv.createTrackbar('11 Convexity%',   WIN, 52, 100, nothing)


def get_params():
    return dict(
        close_size      = max(cv.getTrackbarPos('1 Close size',     WIN), 1),
        close_iter      = max(cv.getTrackbarPos('2 Close iter',     WIN), 1),
        open_size       = max(cv.getTrackbarPos('3 Open size',      WIN), 1),
        open_iter       = max(cv.getTrackbarPos('4 Open iter',      WIN), 1),
        min_area        = cv.getTrackbarPos('5 Min area',           WIN),
        center_zone     = cv.getTrackbarPos('6 Center zone%',       WIN) / 100.0,
        erode_size      = max(cv.getTrackbarPos('7 Erode size',     WIN), 1),
        erode_iter      = max(cv.getTrackbarPos('8 Erode iter',     WIN), 1),
        body_aspect_min = cv.getTrackbarPos('9 Aspect min*10',      WIN) / 10.0,
        body_aspect_max = cv.getTrackbarPos('10 Aspect max*10',     WIN) / 10.0,
        body_convexity  = cv.getTrackbarPos('11 Convexity%',        WIN) / 100.0,
    )


def _grabcut_fg(img):
    h, w = img.shape[:2]
    mx   = max(int(w * GC_RECT_MARGIN_X), 1)
    my   = max(int(h * GC_RECT_MARGIN_Y), 1)
    rect = (mx, my, w - 2 * mx, h - 2 * my)
    mask = np.zeros((h, w), np.uint8)
    bgd  = np.zeros((1, 65), np.float64)
    fgd  = np.zeros((1, 65), np.float64)
    cv.grabCut(img, mask, rect, bgd, fgd, GC_ITER, cv.GC_INIT_WITH_RECT)
    return np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)


def draw_labels(img, p, status, fname, idx, total):
    """Draw current param values and status onto image."""
    h, w = img.shape[:2]
    overlay = img.copy()

    # semi-transparent label box on left
    box_w, box_h = 290, 340
    cv.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    lines = [
        f"[{idx+1}/{total}] {fname}",
        "",
        "-- Mask cleanup --",
        f"  1  Close size : {p['close_size']}",
        f"  2  Close iter : {p['close_iter']}",
        f"  3  Open  size : {p['open_size']}",
        f"  4  Open  iter : {p['open_iter']}",
        "",
        "-- Selection --",
        f"  5  Min area   : {p['min_area']}",
        f"  6  Center zone: {int(p['center_zone']*100)}%",
        "",
        "-- Erosion (leads) --",
        f"  7  Erode size : {p['erode_size']}",
        f"  8  Erode iter : {p['erode_iter']}",
        "",
        "-- Body shape --",
        f"  9  Aspect min : {p['body_aspect_min']:.1f}",
        f"  10 Aspect max : {p['body_aspect_max']:.1f}",
        f"  11 Convexity  : {int(p['body_convexity']*100)}%",
    ]

    y = 18
    for line in lines:
        color = (180, 255, 180) if line.startswith("--") else (220, 220, 220)
        cv.putText(img, line, (6, y), cv.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv.LINE_AA)
        y += 15

    # status bar at bottom
    s_color = (0, 220, 80) if not status.startswith("FAIL") else (0, 60, 255)
    cv.rectangle(img, (0, h - 22), (w, h), (0, 0, 0), -1)
    cv.putText(img, status, (6, h - 7), cv.FONT_HERSHEY_SIMPLEX, 0.45, s_color, 1, cv.LINE_AA)
    cv.putText(img, "R=run  [/]=img  S=print  Q=quit",
               (w - 240, h - 7), cv.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)


def process(img):
    p     = get_params()
    debug = img.copy()
    img_h, img_w = img.shape[:2]

    status     = "RUNNING..."
    eroded_vis = np.zeros((img_h, img_w), dtype=np.uint8)
    fg_vis     = np.zeros((img_h, img_w), dtype=np.uint8)

    try:
        # GrabCut
        fg     = _grabcut_fg(img)
        fg_vis = fg.copy()

        # Mask cleanup
        ck = cv.getStructuringElement(cv.MORPH_ELLIPSE, (p['close_size'], p['close_size']))
        ok = cv.getStructuringElement(cv.MORPH_ELLIPSE, (p['open_size'],  p['open_size']))
        fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, ck, iterations=p['close_iter'])
        fg = cv.morphologyEx(fg, cv.MORPH_OPEN,  ok, iterations=p['open_iter'])

        # Center-zone contour selection
        contours, _ = cv.findContours(fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        margin_x = img_w * (1 - p['center_zone']) / 2
        margin_y = img_h * (1 - p['center_zone']) / 2
        cx_min, cx_max = margin_x, img_w - margin_x
        cy_min, cy_max = margin_y, img_h - margin_y
        cv.rectangle(debug, (int(cx_min), int(cy_min)),
                     (int(cx_max), int(cy_max)), (0, 255, 255), 1)

        centered = []
        for c in contours:
            if cv.contourArea(c) < p['min_area']:
                continue
            x, y, w, h = cv.boundingRect(c)
            if cx_min < x + w/2 < cx_max and cy_min < y + h/2 < cy_max:
                centered.append(c)

        cv.drawContours(debug, centered, -1, (255, 128, 0), 1)

        if not centered:
            raise ValueError("no centered contour")

        best = max(centered, key=cv.contourArea)

        # Erode to strip leads
        full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv.drawContours(full_mask, [best], -1, 255, thickness=cv.FILLED)
        ek     = cv.getStructuringElement(cv.MORPH_ELLIPSE, (p['erode_size'], p['erode_size']))
        eroded = cv.erode(full_mask, ek, iterations=p['erode_iter'])
        eroded_vis = eroded.copy()

        if cv.countNonZero(eroded) == 0:
            raise ValueError("erosion removed contour")

        # Body shape validation
        body_ctrs, _ = cv.findContours(eroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in body_ctrs:
            area = cv.contourArea(c)
            if area < p['min_area'] / 2:
                continue
            _, (bw, bh), _ = cv.minAreaRect(c)
            if min(bw, bh) == 0:
                continue
            aspect = max(bw, bh) / min(bw, bh)
            if not (p['body_aspect_min'] <= aspect <= p['body_aspect_max']):
                continue
            hull = cv.convexHull(c)
            hull_area = cv.contourArea(hull)
            if hull_area > 0 and cv.contourArea(c) / hull_area >= p['body_convexity']:
                candidates.append(c)

        if not candidates:
            raise ValueError("no valid body shape")

        body    = max(candidates, key=cv.contourArea)
        rect_r  = cv.minAreaRect(body)
        box_pts = cv.boxPoints(rect_r).astype(np.int32)
        cv.drawContours(debug, [box_pts], 0, (0, 255, 0), 2)
        _, (bw, bh), angle = rect_r
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        status = f"OK  body:{cv.contourArea(body):.0f}px  aspect:{aspect:.1f}  angle:{angle:.1f}  found:{len(centered)}"

    except Exception as e:
        status = f"FAIL: {e}"

    fname = os.path.basename(all_images[img_index[0]])
    draw_labels(debug, p, status, fname, img_index[0], len(all_images))

    h_img     = debug.shape[0]
    fg_bgr    = cv.resize(cv.cvtColor(fg_vis,     cv.COLOR_GRAY2BGR), (200, h_img))
    erode_bgr = cv.resize(cv.cvtColor(eroded_vis, cv.COLOR_GRAY2BGR), (200, h_img))

    # column headers above side panels
    fg_bgr    = np.vstack([np.full((20, 200, 3), 30, dtype=np.uint8), fg_bgr[:-20]])
    erode_bgr = np.vstack([np.full((20, 200, 3), 30, dtype=np.uint8), erode_bgr[:-20]])
    cv.putText(fg_bgr,    "GrabCut mask", (4, 14), cv.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
    cv.putText(erode_bgr, "After erode",  (4, 14), cv.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

    combined = np.hstack([debug, fg_bgr, erode_bgr])
    cv.imshow(WIN, combined)
    return status


img  = load_img()
last = process(img)

while True:
    key = cv.waitKey(100) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(']'):
        img_index[0] = (img_index[0] + 1) % len(all_images)
        img = load_img()
        print(f"  -> {all_images[img_index[0]]}")
        last = process(img)
    elif key == ord('['):
        img_index[0] = (img_index[0] - 1) % len(all_images)
        img = load_img()
        print(f"  -> {all_images[img_index[0]]}")
        last = process(img)
    elif key == ord('r'):
        print("  Running...")
        last = process(img)
        print(f"  {last}")
    elif key == ord('s'):
        p = get_params()
        print("\n-- Current values --")
        for k, v in p.items():
            print(f"  {k}: {v}")

cv.destroyAllWindows()
