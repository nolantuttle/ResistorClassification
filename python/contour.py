#?------------------------------------------------------------
#? Author: Nolan Tuttle & Mathew Hobson
#? Project: ResistorClassification
#?------------------------------------------------------------

import cv2 as cv
import numpy as np

# ── GrabCut ───────────────────────────────────────────────────────────────────
# Fraction of image width/height reserved as definite background hint.
# Increase if the resistor is small relative to the frame; decrease if it
# fills most of the frame.
RECT_MARGIN_X = 0.40
RECT_MARGIN_Y = 0.40
GC_ITER       = 7       # GrabCut EM iterations

# ── Foreground mask cleanup ───────────────────────────────────────────────────
CLOSE_SIZE = 5          # ellipse kernel for closing (fills gaps in body mask)
CLOSE_ITER = 3
OPEN_SIZE  = 4          # ellipse kernel for opening (removes isolated noise)
OPEN_ITER  = 2

# ── Initial contour selection (includes leads) ────────────────────────────────
MIN_AREA    = 677       # minimum pixel area for candidate contour
CENTER_ZONE = 0.31      # fraction of image considered "center" (x and y)

# ── Body isolation via erosion (strips leads) ─────────────────────────────────
ERODE_SIZE = 8
ERODE_ITER = 1

# ── Body shape validation (after lead removal) ────────────────────────────────
BODY_ASPECT_MIN = 1.1   # min length/width of resistor body
BODY_ASPECT_MAX = 8.0   # max length/width
BODY_CONVEXITY  = 0.52  # min area/convex-hull-area ratio


# ─────────────────────────────────────────────────────────────────────────────

def _grabcut_fg(img, rect_margin_x, rect_margin_y, gc_iter):
    """Return a binary foreground mask using two-pass GrabCut.

    Pass 1: rect initialisation — rough foreground region.
    Pass 2: seed from pass-1 result — eroded core = definite FG,
            dilated inverse = definite BG. Refines edges significantly.
    """
    h, w = img.shape[:2]
    mx   = max(int(w * rect_margin_x), 1)
    my   = max(int(h * rect_margin_y), 1)
    rect = (mx, my, w - 2 * mx, h - 2 * my)

    mask = np.zeros((h, w), np.uint8)
    bgd  = np.zeros((1, 65), np.float64)
    fgd  = np.zeros((1, 65), np.float64)

    # Pass 1 — rect init
    cv.grabCut(img, mask, rect, bgd, fgd, gc_iter, cv.GC_INIT_WITH_RECT)
    fg1 = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)

    # Pass 2 — seed from pass-1 result
    # erode fg1 heavily → definite foreground core
    # dilate complement  → definite background shell
    k_fg = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    k_bg = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,  3))
    sure_fg = cv.erode(fg1,             k_fg, iterations=3)
    sure_bg = cv.dilate(255 - fg1,      k_bg, iterations=2)

    mask2 = np.full((h, w), cv.GC_PR_FGD, dtype=np.uint8)  # prob fg by default
    mask2[sure_bg == 255] = cv.GC_BGD   # definite background
    mask2[sure_fg == 255] = cv.GC_FGD   # definite foreground

    bgd2 = np.zeros((1, 65), np.float64)
    fgd2 = np.zeros((1, 65), np.float64)
    cv.grabCut(img, mask2, None, bgd2, fgd2, gc_iter, cv.GC_INIT_WITH_MASK)

    return np.where((mask2 == 1) | (mask2 == 3), 255, 0).astype(np.uint8)


def _in_center(contour, cx_min, cx_max, cy_min, cy_max):
    x, y, w, h = cv.boundingRect(contour)
    ccx = x + w / 2
    ccy = y + h / 2
    return cx_min < ccx < cx_max and cy_min < ccy < cy_max


def isolate_band_region(image_path,
                        rect_margin_x=RECT_MARGIN_X,
                        rect_margin_y=RECT_MARGIN_Y,
                        gc_iter=GC_ITER,
                        close_size=CLOSE_SIZE,
                        close_iter=CLOSE_ITER,
                        open_size=OPEN_SIZE,
                        open_iter=OPEN_ITER,
                        min_area=MIN_AREA,
                        center_zone=CENTER_ZONE,
                        erode_size=ERODE_SIZE,
                        erode_iter=ERODE_ITER,
                        body_aspect_min=BODY_ASPECT_MIN,
                        body_aspect_max=BODY_ASPECT_MAX,
                        body_convexity=BODY_CONVEXITY):

    img = cv.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    debug_img        = img.copy()
    img_h, img_w     = img.shape[:2]

    # ── 1. GrabCut foreground mask ────────────────────────────────────────────
    fg = _grabcut_fg(img, rect_margin_x, rect_margin_y, gc_iter)

    # ── 2. Morphological cleanup ──────────────────────────────────────────────
    ck = cv.getStructuringElement(cv.MORPH_ELLIPSE, (close_size, close_size))
    ok = cv.getStructuringElement(cv.MORPH_ELLIPSE, (open_size,  open_size))
    fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, ck, iterations=close_iter)
    fg = cv.morphologyEx(fg, cv.MORPH_OPEN,  ok, iterations=open_iter)

    # ── 3. Pick best centered contour (may include leads) ─────────────────────
    contours, _ = cv.findContours(fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours in GrabCut mask")

    margin_x = img_w * (1 - center_zone) / 2
    margin_y = img_h * (1 - center_zone) / 2
    cx_min, cx_max = margin_x, img_w - margin_x
    cy_min, cy_max = margin_y, img_h - margin_y

    centered = [c for c in contours
                if cv.contourArea(c) >= min_area
                and _in_center(c, cx_min, cx_max, cy_min, cy_max)]

    if not centered:
        raise ValueError("No sufficiently large centered contour")

    best = max(centered, key=cv.contourArea)

    # ── 4. Erode to strip leads ───────────────────────────────────────────────
    full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv.drawContours(full_mask, [best], -1, 255, thickness=cv.FILLED)

    ek     = cv.getStructuringElement(cv.MORPH_ELLIPSE, (erode_size, erode_size))
    eroded = cv.erode(full_mask, ek, iterations=erode_iter)

    if cv.countNonZero(eroded) == 0:
        raise ValueError("Erosion removed the entire region — increase erode_size or erode_iter")

    # ── 5. Validate body shape ────────────────────────────────────────────────
    body_ctrs, _ = cv.findContours(eroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not body_ctrs:
        raise ValueError("No body contours after erosion")

    body_candidates = []
    for c in body_ctrs:
        area = cv.contourArea(c)
        if area < min_area / 2:
            continue
        _, (bw, bh), _ = cv.minAreaRect(c)
        if min(bw, bh) == 0:
            continue
        aspect = max(bw, bh) / min(bw, bh)
        if not (body_aspect_min <= aspect <= body_aspect_max):
            continue
        hull      = cv.convexHull(c)
        hull_area = cv.contourArea(hull)
        if hull_area > 0 and area / hull_area >= body_convexity:
            body_candidates.append(c)

    if not body_candidates:
        raise ValueError("No valid body contour after shape filtering")

    body = max(body_candidates, key=cv.contourArea)

    # ── 6. Rotate and crop ────────────────────────────────────────────────────
    rect_rot          = cv.minAreaRect(body)
    center, (bw, bh), angle = rect_rot

    if bw < bh:
        bw, bh = bh, bw
        angle += 90

    M       = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(img, M, (img_w, img_h),
                            flags=cv.INTER_LINEAR,
                            borderMode=cv.BORDER_REPLICATE)

    cx, cy = int(center[0]), int(center[1])
    half_w = int(bw / 2)
    half_h = int(bh / 2)
    x1, x2 = max(cx - half_w, 0), min(cx + half_w, img_w)
    y1, y2 = max(cy - half_h, 0), min(cy + half_h, img_h)

    body_crop = rotated[y1:y2, x1:x2]

    # ── Debug overlay ─────────────────────────────────────────────────────────
    box_pts = cv.boxPoints(rect_rot).astype(np.int32)
    cv.drawContours(debug_img, [box_pts], 0, (0, 255, 0), 2)
    cv.putText(debug_img, "Resistor body",
               (box_pts[1][0], box_pts[1][1] - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return body_crop, debug_img


def preprocess_for_feature_extraction(image_path):
    band_crop, _ = isolate_band_region(image_path)
    return band_crop
