import cv2
import numpy as np
import matplotlib.pyplot as plt

def ideal_lowpass(shape, cutoff):
    P, Q = shape
    u = np.arange(P)
    v = np.arange(Q)
    U, V = np.meshgrid(v - Q//2, u - P//2)
    D = np.sqrt(U**2 + V**2)
    H = np.float32(D <= cutoff)
    return H

def butterworth_lowpass(shape, cutoff, order=2):
    P, Q = shape
    u = np.arange(P)
    v = np.arange(Q)
    U, V = np.meshgrid(v - Q//2, u - P//2)
    D = np.sqrt(U**2 + V**2)
    H = 1 / (1 + (D/cutoff)**(2*order))
    return H

def gaussian_lowpass(shape, cutoff):
    P, Q = shape
    u = np.arange(P)
    v = np.arange(Q)
    U, V = np.meshgrid(v - Q//2, u - P//2)
    D2 = U**2 + V**2
    H = np.exp(-D2 / (2*(cutoff**2)))
    return H

def apply_filter(img, H):
    # Perform DFT
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)

    # Apply filter
    G = dft_shift * H

    # Inverse DFT
    G_ishift = np.fft.ifftshift(G)
    img_back = np.fft.ifft2(G_ishift)
    img_back = np.abs(img_back)
    return img_back

def frequency_domain_lowpass(image_path, cutoff=30):
    # Load grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found. Check the path!")

    img = cv2.resize(img, (256, 256))  # Ensure standard size
    shape = img.shape

    # Create filters
    H_ideal = ideal_lowpass(shape, cutoff)
    H_butterworth = butterworth_lowpass(shape, cutoff, order=2)
    H_gaussian = gaussian_lowpass(shape, cutoff)

    # Apply filters
    img_ideal = apply_filter(img, H_ideal)
    img_butterworth = apply_filter(img, H_butterworth)
    img_gaussian = apply_filter(img, H_gaussian)

    # Show results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    plt.title("Original Image"), plt.axis("off")

    plt.subplot(2, 2, 2), plt.imshow(img_ideal, cmap='gray')
    plt.title("Ideal Low Pass"), plt.axis("off")

    plt.subplot(2, 2, 3), plt.imshow(img_butterworth, cmap='gray')
    plt.title("Butterworth Low Pass"), plt.axis("off")

    plt.subplot(2, 2, 4), plt.imshow(img_gaussian, cmap='gray')
    plt.title("Gaussian Low Pass"), plt.axis("off")

    plt.tight_layout()
    plt.savefig("lowpass_filters_comparison.png")
    plt.show()

# Example usage
if __name__ == "__main__":
    frequency_domain_lowpass("D:\Salman\Semesters\Sem7\Image Processing Lab\Lab Cycle 2\Question10.py\Frozen Rose.jpg", cutoff=40)
