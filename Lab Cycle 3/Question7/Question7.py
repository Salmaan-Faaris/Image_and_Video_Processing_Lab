import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Load the image
image_path = "D:\\Salman\\Semesters\\Sem7\\Image Processing Lab\\Lab Cycle 3\\Question7\\binary_image.png"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Normalize the image
blurred = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)

# Threshold for initial seed selection
_, binary = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)

# Region Growing Implementation
def region_growing(image, seed_points, threshold=10):
    visited = np.zeros_like(image, dtype=bool)
    segmented = np.zeros_like(image, dtype=np.uint8)
    
    for seed in seed_points:
        if visited[seed]:
            continue
        
        queue = deque([seed])
        visited[seed] = True
        region_intensity = float(image[seed])
        region_pixels = [seed]
        
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1] and not visited[nx, ny]:
                    if abs(float(image[nx, ny]) - region_intensity) < threshold:
                        visited[nx, ny] = True
                        queue.append((nx, ny))
                        region_pixels.append((nx, ny))
                        region_intensity = np.mean([image[p] for p in region_pixels])
        
        for (x, y) in region_pixels:
            segmented[x, y] = 255
    
    return segmented

# Choose random seed points from binary regions
seed_points = [(i, j) for i in range(0, binary.shape[0], 30)
               for j in range(0, binary.shape[1], 30)
               if binary[i, j] > 0]

# Apply region growing
segmented = region_growing(blurred, seed_points, threshold=15)

# Display results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Binary (Seed Selection)")
plt.imshow(binary, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Region Grown Segmentation")
plt.imshow(segmented, cmap='gray')
plt.tight_layout()
plt.show()

# Save the output
output_path = "D:\\Salman\\Semesters\\Sem7\\Image Processing Lab\\Lab Cycle 3\\Question7\\RegionGrownSegmentation.jpg"
cv2.imwrite(output_path, segmented)
print(f"✅ Segmented image saved at: {output_path}")
