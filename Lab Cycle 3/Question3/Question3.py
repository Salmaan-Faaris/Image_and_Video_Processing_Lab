import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb_to_hsi(img):
    # Convert image to float and normalize
    img = img.astype(np.float32) / 255.0
    R, G, B = img[:,:,2], img[:,:,1], img[:,:,0]  # OpenCV loads as BGR

    # Intensity
    I = (R + G + B) / 3.0

    # Saturation
    min_val = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 / (R + G + B + 1e-6)) * min_val

    # Hue calculation
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G)**2 + (R - B)*(G - B)) + 1e-6
    theta = np.arccos(num / den)

    H = np.zeros_like(R)
    H[B <= G] = theta[B <= G]
    H[B > G] = (2*np.pi - theta[B > G])
    H = H / (2*np.pi)  # normalize to [0,1]

    return H, S, I

def show_hsi(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image not found!")

    img = cv2.resize(img, (256, 256))  # standard size
    H, S, I = rgb_to_hsi(img)

    # Display results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original RGB Image"), plt.axis("off")

    plt.subplot(2, 2, 2), plt.imshow(H, cmap='hsv')
    plt.title("Hue Component"), plt.axis("off")

    plt.subplot(2, 2, 3), plt.imshow(S, cmap='gray')
    plt.title("Saturation Component"), plt.axis("off")

    plt.subplot(2, 2, 4), plt.imshow(I, cmap='gray')
    plt.title("Intensity Component"), plt.axis("off")

    plt.tight_layout()
    plt.savefig("hsi_components.png")
    plt.show()

# Example usage
if __name__ == "__main__":
    show_hsi("D:\\Salman\\Semesters\\Sem7\\Image Processing Lab\\Lab Cycle 3\\Question3\\Degraded_Image.png")
