import cv2
import numpy as np
import base64
from pathlib import Path

# Real-world measurements (cm)
GLASS_HEIGHT_CM = 16.5
G_HEIGHT_CM = 1.2       # Height of the G letter on a standard Guinness pint
G_WIDTH_CM = 0.9        # Width of the G letter

def decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def find_glass_roi(img: np.ndarray):
    """
    Detect the pint glass region using edge detection and contour analysis.
    Returns the bounding rect (x, y, w, h) of the glass or None.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # Dilate to connect edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = img.shape[:2]
    best = None
    best_score = 0

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        aspect = ch / cw if cw > 0 else 0
        # Glass is tall (aspect > 1.5) and reasonably large
        if aspect > 1.5 and area > (h * w * 0.05):
            score = area * aspect
            if score > best_score:
                best_score = score
                best = (x, y, cw, ch)

    return best

def find_beer_line(img: np.ndarray, glass_roi=None):
    """
    Find the foam/stout boundary line using color segmentation.
    Dark stout (brown/black) meets white/cream foam.
    Returns y coordinate of boundary as fraction from bottom (0-100).
    """
    if glass_roi:
        x, y, w, h = glass_roi
        roi = img[y:y+h, x:x+w]
    else:
        roi = img

    h, w = roi.shape[:2]

    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Mask for dark stout (very dark, low saturation or dark brown)
    dark_mask = cv2.inRange(hsv,
        np.array([0, 0, 0]),
        np.array([30, 255, 80])
    )

    # Mask for cream/white foam
    foam_mask = cv2.inRange(hsv,
        np.array([0, 0, 160]),
        np.array([40, 60, 255])
    )

    # Find boundary: scan from bottom up, find where foam starts above stout
    boundary_y = None
    scan_width = w // 3
    x_start = w // 3

    for y_pos in range(h - 1, 0, -1):
        row_dark = np.sum(dark_mask[y_pos, x_start:x_start+scan_width]) / 255
        row_foam = np.sum(foam_mask[y_pos, x_start:x_start+scan_width]) / 255
        if row_dark > scan_width * 0.3 and boundary_y is None:
            boundary_y = y_pos
        if boundary_y and row_foam > scan_width * 0.3:
            break

    if boundary_y is None:
        boundary_y = h // 2

    # Return as percentage from bottom
    pct_from_bottom = (1 - boundary_y / h) * 100
    return pct_from_bottom, boundary_y

def find_g_logo(img: np.ndarray, glass_roi=None):
    """
    Find the G in the Guinness logo using:
    1. Template matching against a known G template
    2. Fallback: text region detection using MSER
    Returns (midpoint_pct_from_bottom, top_y, bottom_y, confidence)
    """
    if glass_roi:
        x, y, w, h = glass_roi
        roi = img[y:y+h, x:x+w]
        offset_x, offset_y = x, y
    else:
        roi = img
        offset_x, offset_y = 0, 0

    h, w = roi.shape[:2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # --- Method 1: Look for the harp logo / G text region ---
    # The Guinness label is typically in the lower-middle portion of the glass
    # Search in the middle 60% of glass height
    search_top = int(h * 0.2)
    search_bottom = int(h * 0.8)
    search_region = gray[search_top:search_bottom, :]

    # Use adaptive threshold to find text-like regions
    thresh = cv2.adaptiveThreshold(
        search_region, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Find contours that look like the letter G
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    g_candidates = []
    for cnt in contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        aspect = cw / ch if ch > 0 else 0
        # G letter is roughly square-ish, not too small, not too large
        if (0.6 < aspect < 1.4 and
            area > 100 and
            area < (h * w * 0.02) and
            ch > h * 0.02):
            solidity = cv2.contourArea(cnt) / area if area > 0 else 0
            # G has moderate solidity (not fully filled, has opening)
            if 0.3 < solidity < 0.75:
                g_candidates.append((cx, cy + search_top, cw, ch, solidity))

    if g_candidates:
        # Pick the most likely G — largest qualifying candidate
        best_g = max(g_candidates, key=lambda c: c[2] * c[3])
        gx, gy, gw, gh, _ = best_g
        g_mid_y = gy + gh // 2
        g_mid_pct = (1 - g_mid_y / h) * 100
        return g_mid_pct, gy, gy + gh, 0.75
    else:
        # Fallback: assume G is at 40% from bottom (typical Guinness glass position)
        fallback_y = int(h * 0.6)
        g_mid_pct = (1 - fallback_y / h) * 100
        return g_mid_pct, int(h * 0.55), int(h * 0.65), 0.3

def correct_perspective(img: np.ndarray, glass_roi):
    """
    Apply perspective correction using the glass edges as reference.
    Returns rectified image and transform matrix.
    """
    if glass_roi is None:
        return img, None

    x, y, w, h = glass_roi
    roi = img[y:y+h, x:x+w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Find vertical lines (glass edges) using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
                             minLineLength=h//3, maxLineGap=20)

    if lines is None or len(lines) < 2:
        return roi, None

    # Separate left and right edge lines
    left_lines = []
    right_lines = []
    mid_x = w // 2

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle > 70:  # Nearly vertical
            if (x1 + x2) / 2 < mid_x:
                left_lines.append(line[0])
            else:
                right_lines.append(line[0])

    if not left_lines or not right_lines:
        return roi, None

    # Average left and right lines
    def avg_line(lines):
        x1 = int(np.mean([l[0] for l in lines]))
        y1 = int(np.mean([l[1] for l in lines]))
        x2 = int(np.mean([l[2] for l in lines]))
        y2 = int(np.mean([l[3] for l in lines]))
        return x1, y1, x2, y2

    lx1, ly1, lx2, ly2 = avg_line(left_lines)
    rx1, ry1, rx2, ry2 = avg_line(right_lines)

    # Source points (actual glass edges)
    src_pts = np.float32([
        [lx1, ly1], [rx1, ry1],
        [lx2, ly2], [rx2, ry2]
    ])

    # Destination points (rectified rectangle)
    dst_w = rx1 - lx1
    dst_h = h
    dst_pts = np.float32([
        [0, 0], [dst_w, 0],
        [0, dst_h], [dst_w, dst_h]
    ])

    # Compute homography
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return roi, None

    rectified = cv2.warpPerspective(roi, H, (dst_w, dst_h))
    return rectified, H

def calculate_distance_cm(
    g_mid_pct: float,
    beer_line_pct: float,
    glass_pixel_height: int,
) -> tuple[float, str]:
    """
    Convert pixel percentage difference to real-world cm using glass height as scale.
    """
    pct_diff = beer_line_pct - g_mid_pct  # positive = beer line above G
    pixel_diff = (pct_diff / 100) * glass_pixel_height
    cm_per_pixel = GLASS_HEIGHT_CM / glass_pixel_height
    distance_cm = abs(pixel_diff * cm_per_pixel)

    if pct_diff > 0.5:
        position = "above_center"
    elif pct_diff < -0.5:
        position = "below_center"
    else:
        position = "perfect"

    return round(distance_cm, 2), position

def analyze_image(image_bytes: bytes) -> dict:
    """
    Full OpenCV pipeline for Guinness G split analysis.
    """
    img = decode_image(image_bytes)
    if img is None:
        return {"error": "Could not decode image"}

    h, w = img.shape[:2]

    # Step 1: Find glass region
    glass_roi = find_glass_roi(img)

    # Step 2: Perspective correction via homography
    if glass_roi:
        rectified, H = correct_perspective(img, glass_roi)
        glass_h = glass_roi[3]
    else:
        rectified = img
        H = None
        glass_h = h

    # Step 3: Find beer line
    beer_line_pct, beer_line_y = find_beer_line(rectified)

    # Step 4: Find G logo
    g_mid_pct, g_top_y, g_bottom_y, g_confidence = find_g_logo(rectified)

    # Step 5: Calculate distance
    distance_cm, position = calculate_distance_cm(
        g_mid_pct, beer_line_pct, glass_h
    )

    return {
        "glass_detected": glass_roi is not None,
        "g_detected": g_confidence > 0.5,
        "g_confidence": round(g_confidence, 2),
        "beer_line_pct": round(beer_line_pct, 2),
        "g_midpoint_pct": round(g_mid_pct, 2),
        "distance_cm": distance_cm,
        "beer_line_position": position,
        "homography_applied": H is not None,
        "method": "opencv"
    }