import cv2 as cv
import numpy as np

def isolate_band_region(image_path, middle_fraction=0.5):
    img = cv.imread(image_path)
    debug_img = img.copy()

    gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (7,7),0)

    edges = cv.Canny(blurred, threshold1=30, threshold2=100)

    kernal = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
    closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernal)

    # find resistor body contours
    contours, hierarchy = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found in image.")

    # Pick the contour with the largest area — should be the resistor body
    best = max(contours, key=cv.contourArea)

    # --- 3. Get the rotated bounding box ---
    rect = cv.minAreaRect(best)       # (center, (w, h), angle)
    center, (w, h), angle = rect

    # Ensure width > height (resistor is horizontal)
    if w < h:
        w, h = h, w
        angle += 90

    # --- 4. Rotate image to align resistor horizontally ---
    M = cv.getRotationMatrix2D(center, angle, scale=1.0)
    rotated = cv.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv.INTER_LINEAR,
                             borderMode=cv.BORDER_REPLICATE)

    # --- 5. Crop to the rotated bounding box ---
    cx, cy = int(center[0]), int(center[1])
    half_w, half_h = int(w / 2), int(h / 2)

    x1 = max(cx - half_w, 0)
    x2 = min(cx + half_w, rotated.shape[1])
    y1 = max(cy - half_h, 0)
    y2 = min(cy + half_h, rotated.shape[0])

    body_crop = rotated[y1:y2, x1:x2]

    # --- 6. Slice the center fraction (color band region, excludes leads) ---
    bh, bw = body_crop.shape[:2]
    margin = int(bw * (1 - middle_fraction) / 2)
    band_crop = body_crop[:, margin:bw - margin]

    #  Debug: draw contour and crop region on original 
    box_pts = cv.boxPoints(rect).astype(np.int32)
    cv.drawContours(debug_img, [box_pts], 0, (0, 255, 0), 2)
    cv.putText(debug_img, "Resistor body", (box_pts[1][0], box_pts[1][1] - 10),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return band_crop, debug_img


def preprocess_for_feature_extraction(image_path):

    band_crop, _ = isolate_band_region(image_path)
    return band_crop