import cv2 
import numpy as np

def isolate_band_region(image_path, middle_fraction=0.6):
    img = cv2.imread(image_path)
    debug_img = img_copy()

    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7,7),0)

    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernal)



    # find resistor body contour

    contours = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found in image.")

    # Pick the contour with the largest area — should be the resistor body
    best = max(contours, key=cv2.contourArea)

    # --- 3. Get the rotated bounding box ---
    rect = cv2.minAreaRect(best)       # (center, (w, h), angle)
    center, (w, h), angle = rect

    # Ensure width > height (resistor is horizontal)
    if w < h:
        w, h = h, w
        angle += 90

    # --- 4. Rotate image to align resistor horizontally ---
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

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
    box_pts = cv2.boxPoints(rect).astype(np.int32)
    cv2.drawContours(debug_img, [box_pts], 0, (0, 255, 0), 2)
    cv2.putText(debug_img, "Resistor body", (box_pts[1][0], box_pts[1][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return band_crop, debug_img


def preprocess_for_feature_extraction(image_path):

    band_crop, _ = isolate_band_region(image_path)
    return band_crop