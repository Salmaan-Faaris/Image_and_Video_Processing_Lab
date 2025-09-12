import cv2
import numpy as np
import matplotlib.pyplot as plt

def motion_blur_psf(shape, a=0.1, b=0.1, T=1):
    """Generate Motion Blur Point Spread Function (PSF) in frequency domain."""
    M, N = shape
    u = np.arange(M)
    v = np.arange(N)
    U, V = np.meshgrid(u - M//2, v - N//2)
    U = np.fft.ifftshift(U)
    V = np.fft.ifftshift(V)

    # Avoid divide by zero
    pi = np.pi
    temp = (U*a + V*b) * pi
    H = np.ones((M, N), dtype=np.complex64)
    idx = temp != 0
    H[idx] = (T * np.sin(temp[idx]) * np.exp(-1j*temp[idx])) / (pi*(U[idx]*a + V[idx]*b))
    return H

def inverse_filtering(image_path):
    # Load grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found!")

    img = cv2.resize(img, (256, 256))  # standard size

    # --- Step 1: Simulate degradation (motion blur) ---
    H = motion_blur_psf(img.shape, a=0.1, b=0.1, T=1)
    F = np.fft.fft2(img)
    G = H * F
    g = np.abs(np.fft.ifft2(G))  # degraded image

    # --- Step 2: Restore using Inverse Filtering ---
    eps = 1e-6   # avoid division by 0
    F_hat = G / (H + eps)
    f_restored = np.abs(np.fft.ifft2(F_hat))

    # --- Display Results ---
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1), plt.imshow(img, cmap="gray")
    plt.title("Original Image"), plt.axis("off")

    plt.subplot(1, 3, 2), plt.imshow(g, cmap="gray")
    plt.title("Degraded (Blurred)"), plt.axis("off")

    plt.subplot(1, 3, 3), plt.imshow(f_restored, cmap="gray")
    plt.title("Restored (Inverse Filtering)"), plt.axis("off")

    plt.tight_layout()
    plt.savefig("inverse_filtering_result.png")
    plt.show()

# Example usage
if __name__ == "__main__":
    inverse_filtering("D:\\Salman\\Semesters\\Sem7\\Image Processing Lab\\Lab Cycle 3\\Question1\\Degraded_Image.png")
