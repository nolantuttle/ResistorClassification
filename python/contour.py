#?------------------------------------------------------------
#? Author: Nolan Tuttle & Mathew Hobson
#? Project: ResistorClassification
#?------------------------------------------------------------

import cv2 as cv
import numpy as np

CANNY_LOW      = 15
CANNY_HIGH     = 255
LINE_KERNEL    = 3    # morphological open kernel — removes straight lines longer than this
HOUGH_THRESH   = 10   # hough lines threshold
MIN_LINE_LEN   = 10    # hough lines min length
MAX_LINE_GAP   = 65    # hough lines max gap
HOUGH_THICK    = 5     # thickness to erase hough lines from edge map
DILATE_ITER    = 9
ERODE_SIZE     = 10
ERODE_ITER     = 2
ASPECT_MIN     = 1.0
ASPECT_MAX     = 4.0
CONVEXITY_MIN  = 0.45
MIN_AREA       = 1200
CENTER_ZONE    = 0.5

def isolate_band_region(image_path):
    img = cv.imread(image_path)
    debug_img = img.copy()
    img_h, img_w = img.shape[:2]

    gray    = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges   = cv.Canny(blurred, threshold1=CANNY_LOW, threshold2=CANNY_HIGH)

    # step 1: morphological open to remove perfectly straight h/v lines
    lk = max(LINE_KERNEL, 2)
    h_kernel = cv.getStructuringElement(cv.MORPH_RECT, (lk, 1))
    v_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, lk))
    h_lines  = cv.morphologyEx(edges, cv.MORPH_OPEN, h_kernel)
    v_lines  = cv.morphologyEx(edges, cv.MORPH_OPEN, v_kernel)
    edges    = cv.subtract(edges, cv.add(h_lines, v_lines))

    # step 2: hough lines to catch curved/diagonal leads and remaining straight lines
    lines = cv.HoughLinesP(edges, 1, np.pi / 180,
                           threshold=HOUGH_THRESH,
                           minLineLength=MIN_LINE_LEN,
                           maxLineGap=MAX_LINE_GAP)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(edges, (x1, y1), (x2, y2), 0, thickness=HOUGH_THICK)

    # step 3: dilate and close to connect resistor body edges
    kernel  = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dilated = cv.dilate(edges, kernel, iterations=DILATE_ITER)
    closed  = cv.morphologyEx(dilated, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(closed, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in image.")

    # center zone filter
    margin_x = img_w * (1 - CENTER_ZONE) / 2
    margin_y = img_h * (1 - CENTER_ZONE) / 2
    cx_min, cx_max = margin_x, img_w - margin_x
    cy_min, cy_max = margin_y, img_h - margin_y

    filtered = []
    for c in contours:
        area = cv.contourArea(c)
        if area < MIN_AREA:
            continue
        x, y, w, h = cv.boundingRect(c)
        if min(w, h) == 0:
            continue
        ccx = x + w / 2
        ccy = y + h / 2
        if not (cx_min < ccx < cx_max and cy_min < ccy < cy_max):
            continue
        aspect = max(w, h) / min(w, h)
        if not (ASPECT_MIN < aspect < ASPECT_MAX):
            continue
        hull      = cv.convexHull(c)
        hull_area = cv.contourArea(hull)
        if hull_area == 0:
            continue
        if area / hull_area < CONVEXITY_MIN:
            continue
        filtered.append(c)

    if not filtered:
        raise ValueError("No valid contours found.")

    best = max(filtered, key=cv.contourArea)

    # step 4: erode the best contour mask to strip remaining thin leads
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv.drawContours(mask, [best], -1, 255, thickness=cv.FILLED)

    erode_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ERODE_SIZE, ERODE_SIZE))
    eroded       = cv.erode(mask, erode_kernel, iterations=ERODE_ITER)

    if cv.countNonZero(eroded) == 0:
        raise ValueError("Erosion removed entire contour.")

    body_contours, _ = cv.findContours(eroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not body_contours:
        raise ValueError("No contours found after erosion.")

    body           = max(body_contours, key=cv.contourArea)
    rect           = cv.minAreaRect(body)
    center, (w, h), angle = rect

    if w < h:
        w, h   = h, w
        angle += 90

    M       = cv.getRotationMatrix2D(center, angle, scale=1.0)
    rotated = cv.warpAffine(img, M, (img_w, img_h),
                            flags=cv.INTER_LINEAR,
                            borderMode=cv.BORDER_REPLICATE)

    cx, cy         = int(center[0]), int(center[1])
    half_w, half_h = int(w / 2), int(h / 2)

    x1 = max(cx - half_w, 0)
    x2 = min(cx + half_w, img_w)
    y1 = max(cy - half_h, 0)
    y2 = min(cy + half_h, img_h)

    body_crop = rotated[y1:y2, x1:x2]

    box_pts = cv.boxPoints(rect).astype(np.int32)
    cv.drawContours(debug_img, [box_pts], 0, (0, 255, 0), 2)
    cv.putText(debug_img, "Resistor body", (box_pts[1][0], box_pts[1][1] - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return body_crop, debug_img


def preprocess_for_feature_extraction(image_path):
    band_crop, _ = isolate_band_region(image_path)
    return band_crop