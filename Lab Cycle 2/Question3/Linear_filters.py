import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def apply_filter(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv2.filter2D(img, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT)


def average_kernel(k: int) -> np.ndarray:
    return np.ones((k, k), dtype=np.float32) / (k * k)


def gaussian_kernel(k: int, sigma: float = 1.0) -> np.ndarray:
    ax = np.linspace(-(k - 1) / 2., (k - 1) / 2., k)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel.astype(np.float32)


def sobel_kernels():
    kx = np.array([[ -1, 0, 1],
                   [ -2, 0, 2],
                   [ -1, 0, 1]], dtype=np.float32)
    ky = np.array([[ -1, -2, -1],
                   [  0,  0,  0],
                   [  1,  2,  1]], dtype=np.float32)
    return kx, ky


def prewitt_kernels():
    kx = np.array([[ -1, 0, 1],
                   [ -1, 0, 1],
                   [ -1, 0, 1]], dtype=np.float32)
    ky = np.array([[ -1, -1, -1],
                   [  0,  0,  0],
                   [  1,  1,  1]], dtype=np.float32)
    return kx, ky


def roberts_kernels():
    kx = np.array([[1, 0],
                   [0, -1]], dtype=np.float32)
    ky = np.array([[0, 1],
                   [-1, 0]], dtype=np.float32)
    return kx, ky


def scharr_kernels():
    kx = np.array([[ -3, 0, 3],
                   [ -10, 0, 10],
                   [ -3, 0, 3]], dtype=np.float32)
    ky = np.array([[ -3, -10, -3],
                   [  0,   0,  0],
                   [  3,  10,  3]], dtype=np.float32)
    return kx, ky


def laplacian_kernel():
    return np.array([[0,  1, 0],
                     [1, -4, 1],
                     [0,  1,  0]], dtype=np.float32)


def high_pass_kernel():
    avg = average_kernel(3)
    identity = np.zeros_like(avg)
    identity[1, 1] = 1
    return identity - avg


def unsharp_mask(img: np.ndarray, kernel_size=5, sigma=1.0, amount=1.0, threshold=0) -> np.ndarray:
    blurred = apply_filter(img, gaussian_kernel(kernel_size, sigma))
    diff = cv2.subtract(img, blurred)
    sharpened = cv2.addWeighted(img, 1.0, diff, amount, 0)
    if threshold > 0:
        low_contrast_mask = np.absolute(diff) < threshold
        np.copyto(sharpened, img, where=low_contrast_mask)
    return sharpened


def magnitude_from_gradients(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    mag = np.sqrt(np.square(gx.astype(np.float32)) + np.square(gy.astype(np.float32)))
    mag = np.clip(mag, 0, 255)
    return mag.astype(np.uint8)


def to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def make_montage(results: dict, cols: int = 4) -> np.ndarray:
    rows = int(np.ceil(len(results) / cols))
    h, w = next(iter(results.values())).shape
    montage = np.zeros((rows * h, cols * w), dtype=np.uint8)
    for idx, out in enumerate(results.values()):
        r = idx // cols
        c = idx % cols
        montage[r*h:(r+1)*h, c*w:(c+1)*w] = out
    return montage


def demo_and_save_montage(image_path: str, out_path: str):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    img = to_gray(img_bgr)

    results = {}
    results['Original'] = img
    results['Average_5x5'] = apply_filter(img, average_kernel(5))
    results['Gaussian_7x7_sigma1.5'] = apply_filter(img, gaussian_kernel(7, sigma=1.5))
    results['Laplacian'] = apply_filter(img, laplacian_kernel())
    results['Highpass'] = apply_filter(img, high_pass_kernel())
    results['Unsharp'] = unsharp_mask(img, kernel_size=5, sigma=1.0, amount=1.5)

    sx, sy = sobel_kernels()
    gx = apply_filter(img, sx); gy = apply_filter(img, sy)
    results['Sobel'] = magnitude_from_gradients(gx, gy)

    px, py = prewitt_kernels()
    gx = apply_filter(img, px); gy = apply_filter(img, py)
    results['Prewitt'] = magnitude_from_gradients(gx, gy)

    rx, ry = roberts_kernels()
    gx = apply_filter(img, rx); gy = apply_filter(img, ry)
    results['Roberts'] = magnitude_from_gradients(gx, gy)

    schx, schy = scharr_kernels()
    gx = apply_filter(img, schx); gy = apply_filter(img, schy)
    results['Scharr'] = magnitude_from_gradients(gx, gy)

    montage = make_montage(results, cols=4)
    cv2.imwrite(out_path, montage)
    print(f"Saved montage to: {out_path}")

    plt.imshow(montage, cmap='gray')
    plt.title("Montage of Filters")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create and save montage of linear spatial filters.")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("-o", "--out", default="output.png", help="Output montage filename (PNG)")
    args = parser.parse_args()

    demo_and_save_montage(args.image, args.out)
