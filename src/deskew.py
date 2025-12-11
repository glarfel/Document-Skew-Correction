import cv2
import numpy as np
import os


def load_image(path: str):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def preprocess_for_lines(gray):
    """Blur + Canny edges."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges


def estimate_angle_hough(gray, debug=False):
  
    edges = preprocess_for_lines(gray)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=100,
        maxLineGap=10,
    )

    if lines is None or len(lines) == 0:
        if debug:
            print("No lines found. Returning 0.")
        return 0.0

    angles = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))

        if angle < -90:
            angle += 180
        if angle >= 90:
            angle -= 180

        if -45 < angle < 45:
            angles.append(angle)

    if len(angles) == 0:
        if debug:
            print("No near-horizontal lines. Returning 0.")
        return 0.0

    hist, bin_edges = np.histogram(angles, bins=90, range=(-45, 45))
    max_bin_index = np.argmax(hist)
    bin_start = bin_edges[max_bin_index]
    bin_end = bin_edges[max_bin_index + 1]
    dominant_angle = (bin_start + bin_end) / 2.0

    if debug:
        print(f"Collected {len(angles)} angles.")
        print(f"Dominant angle (deg): {dominant_angle:.2f}")

    return dominant_angle


def rotate_image_keep_bounds(img, angle_deg):

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def deskew_image(img, debug=False):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    est_angle = estimate_angle_hough(gray, debug=debug)

    cand1 = rotate_image_keep_bounds(img, -est_angle)
    gray1 = cv2.cvtColor(cand1, cv2.COLOR_BGR2GRAY)
    resid1 = abs(estimate_angle_hough(gray1, debug=False))

    cand2 = rotate_image_keep_bounds(img, est_angle)
    gray2 = cv2.cvtColor(cand2, cv2.COLOR_BGR2GRAY)
    resid2 = abs(estimate_angle_hough(gray2, debug=False))

    if resid1 <= resid2:
        corrected = cand1
        chosen_rot = -est_angle
    else:
        corrected = cand2
        chosen_rot = est_angle

    if debug:
        print(f"Initial estimate: {est_angle:.2f} deg")
        print(f"Residual after -est: {resid1:.2f} deg")
        print(f"Residual after +est: {resid2:.2f} deg")
        print(f"Chosen rotation to apply: {chosen_rot:.2f} deg")

    return corrected, est_angle
