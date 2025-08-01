import numpy as np
import cv2
import os
from typing import Tuple


def read_mask(path: str) -> np.ndarray:
    """
    Reads an ASCII mask file. Expect whitespace-separated numbers; blank lines and
    lines starting with '#' are ignored.
    """
    with open(path, 'r') as f:
        lines = []
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            vals = [float(p) for p in parts]
            lines.append(vals)
    if not lines:
        raise ValueError(f"No valid data found in mask file {path}")
    mat = np.array(lines, dtype=np.float32)
    h, w = mat.shape
    if h % 2 == 0 or w % 2 == 0:
        raise ValueError("Mask must have odd dimensions.")
    return mat


def pad_reflect(img: np.ndarray, pad_y: int, pad_x: int) -> np.ndarray:
    return np.pad(img, ((pad_y, pad_y), (pad_x, pad_x)), mode='reflect')


def correlate_single_channel(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Correlation: slide kernel over image without flipping.
    """
    kh, kw = kernel.shape
    pad_y = kh // 2
    pad_x = kw // 2
    img_p = pad_reflect(img, pad_y, pad_x)
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            region = img_p[i:i+kh, j:j+kw]
            out[i, j] = np.sum(region * kernel)
    # Clip to valid range after returning
    return out


def convolve_single_channel(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolution: flip kernel both axes, then correlate.
    """
    flipped = np.flip(np.flip(kernel, axis=0), axis=1)
    return correlate_single_channel(img, flipped)


def apply_filter(img: np.ndarray, kernel: np.ndarray, mode: str) -> np.ndarray:
    """
    mode: 'correlation' or 'convolution'
    Works on grayscale or color (per-channel).
    """
    if mode not in ('correlation', 'convolution'):
        raise ValueError("mode must be 'correlation' or 'convolution'")

    single_op = convolve_single_channel if mode == 'convolution' else correlate_single_channel

    if img.ndim == 2:  # grayscale
        out = single_op(img, kernel)
    else:
        chans = []
        for c in range(img.shape[2]):
            chan = single_op(img[..., c], kernel)
            chans.append(chan)
        out = np.stack(chans, axis=2)
    # Normalize or clip to 0-255
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)


def write_mask_file(size: int, path: str):
    """Writes an averaging mask of given odd size to path."""
    if size % 2 == 0:
        raise ValueError("Size must be odd.")
    val = 1.0 / (size * size)
    with open(path, 'w') as f:
        f.write(f"# {size}x{size} averaging kernel\n")
        for _ in range(size):
            row = " ".join(f"{val:.8f}" for _ in range(size))
            f.write(row + "\n")


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def demo(image_path: str, output_dir: str):
    """
    Generates the three averaging masks, applies correlation and convolution,
    and saves ONLY the montage image per mask+mode combination.
    """
    ensure_dir(output_dir)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load image at {image_path}")
    # Convert to grayscale for simplicity (you can also test color by commenting next line)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    mask_sizes = [3, 7, 11]
    for size in mask_sizes:
        mask_fname = os.path.join(output_dir, f"avg_{size}x{size}.txt")
        write_mask_file(size, mask_fname)
        kernel = read_mask(mask_fname)

        for mode in ("correlation", "convolution"):
            filtered = apply_filter(img, kernel, mode)

            # Build montage: [original | filtered]
            h, w = img.shape
            montage = np.zeros((h, w * 2), dtype=np.uint8)
            montage[:, :w] = img
            montage[:, w:] = filtered

            out_name = f"montage_{size}x{size}_{mode}.png"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, montage)
            print(f"Saved {mode} montage for {size}x{size} to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Image convolution and correlation with ASCII mask.")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("-o", "--outdir", default="results", help="Directory to save outputs")
    args = parser.parse_args()

    demo(args.image, args.outdir)
