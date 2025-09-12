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

    pi = np.pi
    temp = (U*a + V*b) * pi
    H = np.ones((M, N), dtype=np.complex64)
    idx = temp != 0
    H[idx] = (T * np.sin(temp[idx]) * np.exp(-1j*temp[idx])) / (pi*(U[idx]*a + V[idx]*b))
    return H

def wiener_filter(G, H, K=0.01, Snn=None, Sff=None):
    """Wiener filter implementation.
       If Snn and Sff are provided, use autocorrelation-based Wiener filtering.
       Otherwise, use constant K Wiener filtering.
    """
    H_conj = np.conj(H)
    H_abs2 = np.abs(H)**2
    
    if Snn is not None and Sff is not None:
        # Auto-correlation based Wiener filtering
        NSR = Snn / (Sff + 1e-6)
    else:
        # Constant ratio Wiener filtering
        NSR = K
    
    W_filter = H_conj / (H_abs2 + NSR)
    F_hat = W_filter * G
    f_restored = np.abs(np.fft.ifft2(F_hat))
    return f_restored

def wiener_filtering(image_path):
    # Load grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found!")

    img = cv2.resize(img, (256, 256))  # standard size

    # --- Step 1: Simulate degradation (motion blur + noise) ---
    H = motion_blur_psf(img.shape, a=0.1, b=0.1, T=1)
    F = np.fft.fft2(img)
    G = H * F

    # Add Gaussian noise
    noise = np.random.normal(0, 20, img.shape)
    g_noisy = np.abs(np.fft.ifft2(G)) + noise
    G = np.fft.fft2(g_noisy)

    # --- Step 2a: Constant Ratio Wiener Filtering ---
    restored_const = wiener_filter(G, H, K=0.01)

    # --- Step 2b: Autocorrelation-based Wiener Filtering ---
    Sff = np.abs(F)**2     # Power spectrum of original
    Snn = np.abs(np.fft.fft2(noise))**2  # Power spectrum of noise
    restored_auto = wiener_filter(G, H, Snn=Snn, Sff=Sff)

    # --- Display Results ---
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1), plt.imshow(img, cmap="gray")
    plt.title("Original Image"), plt.axis("off")

    plt.subplot(2, 2, 2), plt.imshow(g_noisy, cmap="gray")
    plt.title("Degraded (Blur + Noise)"), plt.axis("off")

    plt.subplot(2, 2, 3), plt.imshow(restored_const, cmap="gray")
    plt.title("Restored (Wiener, Constant Ratio)"), plt.axis("off")

    plt.subplot(2, 2, 4), plt.imshow(restored_auto, cmap="gray")
    plt.title("Restored (Wiener, Autocorrelation)"), plt.axis("off")

    plt.tight_layout()
    plt.savefig("wiener_filtering_results.png")
    plt.show()

# Example usage
if __name__ == "__main__":
    wiener_filtering("D:\\Salman\\Semesters\\Sem7\\Image Processing Lab\\Lab Cycle 3\\Question2\\Degraded_Image.png")
