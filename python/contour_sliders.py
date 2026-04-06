import cv2 as cv
import numpy as np
import os
import sys

IMAGE_DIR = sys.argv[1] if len(sys.argv) > 1 else 'archive'
WIN       = 'Tune Contours'

def nothing(x): pass

all_images = []
for root, dirs, files in os.walk(IMAGE_DIR):
    for f in files:
        if f.endswith('.jpg'):
            all_images.append(os.path.join(root, f))
all_images.sort()

assert len(all_images) > 0, f"No jpg images found in {IMAGE_DIR}"
print(f"Found {len(all_images)} images.")

img_index = [0]

def load_img():
    return cv.imread(all_images[img_index[0]])

cv.namedWindow(WIN, cv.WINDOW_NORMAL)
cv.resizeWindow(WIN, 1200, 700)

cv.createTrackbar('Canny low',       WIN,  30,  255, nothing)
cv.createTrackbar('Canny high',      WIN, 100,  255, nothing)
cv.createTrackbar('Line kernel',     WIN,  40,  150, nothing)
cv.createTrackbar('Hough thresh',    WIN, 100,  300, nothing)
cv.createTrackbar('Min line len',    WIN,  50,  300, nothing)
cv.createTrackbar('Max line gap',    WIN,  10,  100, nothing)
cv.createTrackbar('Hough thick',     WIN,   5,   20, nothing)
cv.createTrackbar('Dilate iter',     WIN,   2,   10, nothing)
cv.createTrackbar('Erode size',      WIN,  15,   60, nothing)
cv.createTrackbar('Erode iter',      WIN,   2,   10, nothing)
cv.createTrackbar('Aspect min x10',  WIN,  15,  100, nothing)
cv.createTrackbar('Aspect max x10',  WIN,  60,  100, nothing)
cv.createTrackbar('Convexity x100',  WIN,  70,  100, nothing)
cv.createTrackbar('Min area',        WIN, 500, 9000, nothing)
cv.createTrackbar('Center zone x10', WIN,   6,   10, nothing)

def process(img):
    canny_low    = cv.getTrackbarPos('Canny low',       WIN)
    canny_high   = cv.getTrackbarPos('Canny high',      WIN)
    line_kernel  = max(cv.getTrackbarPos('Line kernel', WIN), 2)
    hough_thresh = cv.getTrackbarPos('Hough thresh',    WIN)
    min_line_len = cv.getTrackbarPos('Min line len',    WIN)
    max_line_gap = cv.getTrackbarPos('Max line gap',    WIN)
    hough_thick  = cv.getTrackbarPos('Hough thick',     WIN)
    dilate_iter  = max(cv.getTrackbarPos('Dilate iter', WIN), 1)
    erode_size   = max(cv.getTrackbarPos('Erode size',  WIN), 1)
    erode_iter   = max(cv.getTrackbarPos('Erode iter',  WIN), 1)
    aspect_min   = cv.getTrackbarPos('Aspect min x10',  WIN) / 10.0
    aspect_max   = cv.getTrackbarPos('Aspect max x10',  WIN) / 10.0
    convexity    = cv.getTrackbarPos('Convexity x100',  WIN) / 100.0
    min_area     = cv.getTrackbarPos('Min area',        WIN)
    center_zone  = cv.getTrackbarPos('Center zone x10', WIN) / 10.0

    debug  = img.copy()
    img_h, img_w = img.shape[:2]

    gray    = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges   = cv.Canny(blurred, canny_low, canny_high)

    # morphological line removal
    h_kernel = cv.getStructuringElement(cv.MORPH_RECT, (line_kernel, 1))
    v_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, line_kernel))
    h_lines  = cv.morphologyEx(edges, cv.MORPH_OPEN, h_kernel)
    v_lines  = cv.morphologyEx(edges, cv.MORPH_OPEN, v_kernel)
    edges    = cv.subtract(edges, cv.add(h_lines, v_lines))

    # hough line removal
    lines = cv.HoughLinesP(edges, 1, np.pi / 180,
                           threshold=hough_thresh,
                           minLineLength=min_line_len,
                           maxLineGap=max_line_gap)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(edges, (x1, y1), (x2, y2), 0, thickness=hough_thick)

    kernel  = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dilated = cv.dilate(edges, kernel, iterations=dilate_iter)
    closed  = cv.morphologyEx(dilated, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(closed, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    margin_x = img_w * (1 - center_zone) / 2
    margin_y = img_h * (1 - center_zone) / 2
    cx_min, cx_max = margin_x, img_w - margin_x
    cy_min, cy_max = margin_y, img_h - margin_y

    cv.rectangle(debug, (int(cx_min), int(cy_min)), (int(cx_max), int(cy_max)), (0, 255, 255), 1)

    filtered = []
    for c in contours:
        area = cv.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv.boundingRect(c)
        if min(w, h) == 0:
            continue
        ccx = x + w / 2
        ccy = y + h / 2
        if not (cx_min < ccx < cx_max and cy_min < ccy < cy_max):
            continue
        aspect = max(w, h) / min(w, h)
        if not (aspect_min < aspect < aspect_max):
            continue
        hull      = cv.convexHull(c)
        hull_area = cv.contourArea(hull)
        if hull_area == 0:
            continue
        if area / hull_area < convexity:
            continue
        filtered.append(c)

    cv.drawContours(debug, filtered, -1, (255, 0, 0), 1)

    status     = "NO VALID CONTOURS"
    eroded_vis = np.zeros((img_h, img_w), dtype=np.uint8)

    if filtered:
        best = max(filtered, key=cv.contourArea)
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv.drawContours(mask, [best], -1, 255, thickness=cv.FILLED)

        erode_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (erode_size, erode_size))
        eroded       = cv.erode(mask, erode_kernel, iterations=erode_iter)
        eroded_vis   = eroded.copy()

        if cv.countNonZero(eroded) > 0:
            body_contours, _ = cv.findContours(eroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if body_contours:
                body    = max(body_contours, key=cv.contourArea)
                rect    = cv.minAreaRect(body)
                box_pts = cv.boxPoints(rect).astype(np.int32)
                cv.drawContours(debug, [box_pts], 0, (0, 255, 0), 2)
                _, (bw, bh), angle = rect
                status = f"pre:{cv.contourArea(best):.0f} post:{cv.contourArea(body):.0f} angle:{angle:.1f} found:{len(filtered)}"
            else:
                status = "EROSION KILLED CONTOUR"
        else:
            status = "EROSION KILLED CONTOUR"

    fname = os.path.basename(all_images[img_index[0]])
    cv.putText(debug, f"[{img_index[0]+1}/{len(all_images)}] {fname}",
               (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv.putText(debug, status,
               (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 100), 1)

    h_img      = debug.shape[0]
    edge_vis   = cv.resize(cv.cvtColor(closed,     cv.COLOR_GRAY2BGR), (220, h_img))
    erode_view = cv.resize(cv.cvtColor(eroded_vis, cv.COLOR_GRAY2BGR), (220, h_img))
    combined   = np.hstack([debug, edge_vis, erode_view])
    cv.imshow(WIN, combined)

print("[ / ] — prev/next    S — print values    Q — quit")

img = load_img()
while True:
    process(img)
    key = cv.waitKey(50) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(']'):
        img_index[0] = (img_index[0] + 1) % len(all_images)
        img = load_img()
        print(f"  → {all_images[img_index[0]]}")
    elif key == ord('['):
        img_index[0] = (img_index[0] - 1) % len(all_images)
        img = load_img()
        print(f"  → {all_images[img_index[0]]}")
    elif key == ord('s'):
        print("\n── Current values ──")
        for name in ['Canny low','Canny high','Line kernel','Hough thresh',
                     'Min line len','Max line gap','Hough thick','Dilate iter',
                     'Erode size','Erode iter','Aspect min x10','Aspect max x10',
                     'Convexity x100','Min area','Center zone x10']:
            print(f"  {name}: {cv.getTrackbarPos(name, WIN)}")

cv.destroyAllWindows()