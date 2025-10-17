import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def segment_blobs(image_path, output_path="segmented_blobs.png"):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image not found. Check the path.")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Otsu's thresholding (automatically selects optimal threshold)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optionally, invert the image if blobs are dark on light background
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    # Find contours (each contour corresponds to a blob)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours (segment blobs)
    segmented = img.copy()
    cv2.drawContours(segmented, contours, -1, (0, 255, 0), 2)

    # Save output
    cv2.imwrite(output_path, segmented)
    print(f"✅ Segmented blob image saved as: {os.path.abspath(output_path)}")

    # Display results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(binary, cmap='gray')
    plt.title("Binary Thresholded Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    plt.title("Segmented Blobs (Contours)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# ---- Main Execution ----
image_path = "D:\Salman\Semesters\Sem7\Image Processing Lab\Lab Cycle 3\Question4\Frozen Rose.jpg"
segment_blobs(image_path, output_path="segmented_blobs.png")
