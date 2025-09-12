import cv2
import numpy as np
import matplotlib.pyplot as plt

def laplacian_sharpen(image_path, output_path="laplacian_sharpened.png"):
    # Read the image (BGR)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError("Image not found. Check the path.")

    # Convert to RGB (for display)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Apply Laplacian filter
    laplacian = cv2.Laplacian(img_rgb, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)

    # Sharpened = Original + Laplacian
    sharpened = cv2.addWeighted(img_rgb, 1.0, laplacian, 1.0, 0)

    # Save result
    sharpened_bgr = cv2.cvtColor(sharpened, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, sharpened_bgr)

    # Display original vs sharpened
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(sharpened)
    plt.title("Laplacian Sharpened")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Example usage
image_path = r"D:\\Salman\\Semesters\\Sem7\\Image Processing Lab\\Lab Cycle 2\\Question8\\Frozen Rose.jpg"  
laplacian_sharpen(image_path, output_path="sharpened_result.png")
