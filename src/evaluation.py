import csv
import os
import numpy as np
import cv2

from deskew import load_image, deskew_image

SYNTHETIC_DIR = os.path.join("data", "synthetic")
ANNOTATIONS_PATH = os.path.join(SYNTHETIC_DIR, "annotations.csv")


def load_annotations(csv_path):
    records = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["image"]:
                continue
            records.append(row)
    return records


def evaluate():
    records = load_annotations(ANNOTATIONS_PATH)
    errors = []

    for r in records:
        img_name = r["image"]
        true_angle = float(r["true_angle_deg"])

        img_path = os.path.join(SYNTHETIC_DIR, img_name)
        img = load_image(img_path)

        corrected, est_angle = deskew_image(img, debug=False)

        e1 = abs(est_angle - true_angle)
        e2 = abs(est_angle + true_angle)
        error = min(e1, e2)

        errors.append(error)

    errors = np.array(errors)
    print(f"Number of images: {len(errors)}")
    print(f"Mean absolute error (deg): {errors.mean():.3f}")
    print(f"Median absolute error (deg): {np.median(errors):.3f}")
    print(f"Max absolute error (deg): {errors.max():.3f}")
    print(f"Within 1 degree: {(errors < 1).mean() * 100:.1f}%")
    print(f"Within 2 degrees: {(errors < 2).mean() * 100:.1f}%")


if __name__ == "__main__":
    evaluate()
