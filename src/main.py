import argparse
import os
import cv2

from deskew import load_image, deskew_image


def main():
    parser = argparse.ArgumentParser(description="Document skew correction demo")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--save_dir", default=None, help="Where to save result")
    parser.add_argument("--debug", action="store_true", help="Print debug info")
    args = parser.parse_args()

    img = load_image(args.image)
    corrected, est_angle = deskew_image(img, debug=args.debug)

    print(f"Estimated angle: {est_angle:.2f} degrees")

    # Save output
    if args.save_dir is None:
        dirname, fname = os.path.split(args.image)
        name, ext = os.path.splitext(fname)
        out_path = os.path.join(dirname, f"{name}_deskewed{ext}")
    else:
        os.makedirs(args.save_dir, exist_ok=True)
        fname = os.path.basename(args.image)
        name, ext = os.path.splitext(fname)
        out_path = os.path.join(args.save_dir, f"{name}_deskewed{ext}")

    cv2.imwrite(out_path, corrected)
    print(f"Saved corrected image to: {out_path}")


if __name__ == "__main__":
    main()
