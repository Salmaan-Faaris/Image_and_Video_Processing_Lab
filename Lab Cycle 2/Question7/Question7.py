import cv2
import numpy as np
import matplotlib.pyplot as plt

def smoothen_rgb(image_path, kernel_size=5, output_path="smoothed.png"):
    # Read the image (BGR)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError("Image not found. Check the path.")
    
    # Convert to RGB for display
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Create averaging filter kernel
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    
    # Apply linear spatial filter
    smoothed = cv2.filter2D(img_rgb, -1, kernel)
    
    # Save result
    smoothed_bgr = cv2.cvtColor(smoothed, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, smoothed_bgr)
    
    # Display original vs smoothed
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(img_rgb)
    plt.title("Original")
    plt.axis("off")
    
    plt.subplot(1,2,2)
    plt.imshow(smoothed)
    plt.title(f"Smoothed ({kernel_size}x{kernel_size})")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

# Example usage
image_path = "Frozen Rose.jpg"   # Change this to your image path
smoothen_rgb(image_path, kernel_size=7, output_path="smoothed_result.png")
