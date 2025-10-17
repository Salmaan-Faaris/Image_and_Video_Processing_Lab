import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def detect_lines_hough(image_path, output_path="hough_lines_result.png"):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image not found. Check the path.")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert to binary (thresholding)
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    # Edge detection
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)

    # Apply Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=80,
                            minLineLength=30, maxLineGap=10)

    # Draw lines on the original image
    result = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Save output
    cv2.imwrite(output_path, result)
    print(f"✅ Hough line detection result saved as: {os.path.abspath(output_path)}")

    # Display results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(edges, cmap='gray')
    plt.title("Edge Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Detected Line Segments (Hough Transform)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ---- Main Execution ----
image_path = "D:\Salman\Semesters\Sem7\Image Processing Lab\Lab Cycle 3\Question4\Frozen Rose.jpg"
detect_lines_hough(image_path, output_path="hough_lines_result.png")
