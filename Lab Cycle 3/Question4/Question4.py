import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb_to_hsi(img):
    img = img.astype(np.float32) / 255
    R, G, B = cv2.split(img)

    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G)**2 + (R - B)*(G - B)) + 1e-6
    theta = np.arccos(num / den)

    H = np.where(B <= G, theta, 2 * np.pi - theta)
    H = H / (2 * np.pi)

    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 / (R + G + B + 1e-6)) * min_rgb
    I = (R + G + B) / 3

    return H, S, I

def hsi_to_rgb(H, S, I):
    H = H * 2 * np.pi
    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)

    # Sector 0°–120°
    idx = (H >= 0) & (H < 2*np.pi/3)
    B[idx] = I[idx] * (1 - S[idx])
    R[idx] = I[idx] * (1 + (S[idx] * np.cos(H[idx])) / (np.cos(np.pi/3 - H[idx]) + 1e-6))
    G[idx] = 3*I[idx] - (R[idx] + B[idx])

    # Sector 120°–240°
    idx = (H >= 2*np.pi/3) & (H < 4*np.pi/3)
    H2 = H[idx] - 2*np.pi/3
    R[idx] = I[idx] * (1 - S[idx])
    G[idx] = I[idx] * (1 + (S[idx] * np.cos(H2)) / (np.cos(np.pi/3 - H2) + 1e-6))
    B[idx] = 3*I[idx] - (R[idx] + G[idx])

    # Sector 240°–360°
    idx = (H >= 4*np.pi/3) & (H < 2*np.pi)
    H3 = H[idx] - 4*np.pi/3
    G[idx] = I[idx] * (1 - S[idx])
    B[idx] = I[idx] * (1 + (S[idx] * np.cos(H3)) / (np.cos(np.pi/3 - H3) + 1e-6))
    R[idx] = 3*I[idx] - (G[idx] + B[idx])

    rgb = cv2.merge((R, G, B))
    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)

def histogram_equalize_intensity(img_path, output_path="equalized_result.png"):
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError("Image not found. Check the path.")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert RGB → HSI
    H, S, I = rgb_to_hsi(img_rgb)

    # Equalize Intensity
    I_eq = cv2.equalizeHist((I * 255).astype(np.uint8))
    I_eq = I_eq.astype(np.float32) / 255

    # Reconstruct new RGB image
    new_rgb = hsi_to_rgb(H, S, I_eq)

    # Save output
    new_rgb_bgr = cv2.cvtColor(new_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, new_rgb_bgr)

    print(f"✅ Equalized image saved as: {output_path}")

    # Display results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original RGB")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(I, cmap='gray')
    plt.title("Original Intensity")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(new_rgb)
    plt.title("Histogram Equalized RGB")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Example Usage
image_path = "D:\Salman\Semesters\Sem7\Image Processing Lab\Lab Cycle 3\Question4\Frozen Rose.jpg"
histogram_equalize_intensity(image_path, output_path="equalized_result.png")
