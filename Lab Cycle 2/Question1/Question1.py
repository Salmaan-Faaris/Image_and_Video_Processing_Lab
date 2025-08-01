import cv2
import numpy as np
import argparse
import os

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img

def match_size(src, target):
    """Resize target to match src's dimensions."""
    return cv2.resize(target, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_AREA)

def safe_divide(a, b):
    """Divide with handling for zeros in denominator."""
    # cv2.divide already handles division by zero by setting result to max value; we can also do manual:
    b_safe = b.astype(np.float32)
    b_safe[b_safe == 0] = 1e-6  # avoid zero
    return (a.astype(np.float32) / b_safe).clip(0, 255).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description="Perform add/subtract/multiply/divide between two images.")
    parser.add_argument("image1", help="Path to first image")
    parser.add_argument("image2", help="Path to second image")
    parser.add_argument("--output-dir", "-o", default="results", help="Directory to save output images")
    parser.add_argument("--show", "-s", action="store_true", help="Show results in windows")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for weighted addition (only affects add): alpha for image1, 1-alpha for image2")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    img1 = load_image(args.image1)
    img2 = load_image(args.image2)
    img2 = match_size(img1, img2)

    # Addition (normal and weighted)
    added = cv2.add(img1, img2)  # saturating add
    weighted = cv2.addWeighted(img1, args.alpha, img2, 1 - args.alpha, 0)

    # Subtraction (img1 - img2 and absolute difference)
    sub = cv2.subtract(img1, img2)
    absdiff = cv2.absdiff(img1, img2)

    # Multiplication
    multiplied = cv2.multiply(img1, img2)
    # Normalize multiplied to avoid overflow (optional): scale down if values exceed 255
    # multiplied = cv2.normalize(multiplied, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Division (img1 / img2)
    divided = safe_divide(img1, img2)

    outputs = {
        "added": added,
        "subtracted": sub,
        "multiplied": multiplied,
        "divided": divided,
    }

    for name, img in outputs.items():
        out_path = os.path.join(args.output_dir, f"{name}.png")
        cv2.imwrite(out_path, img)
        print(f"Saved {name} -> {out_path}")
        if args.show:
            cv2.imshow(name, img)

    if args.show:
        print("Press any key in any image window to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
