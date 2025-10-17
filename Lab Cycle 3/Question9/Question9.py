import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------
# Load the input image
# ----------------------------------
image_path = "D:\Salman\Semesters\Sem7\Image Processing Lab\Lab Cycle 3\Question4\Frozen Rose.jpg"
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError("❌ Image not found. Check the path!")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ----------------------------------
# Step 1: Threshold to create binary image (simulate blobs)
# ----------------------------------
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# ----------------------------------
# Step 2: Noise removal using Morphology
# ----------------------------------
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# Sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# ----------------------------------
# Step 3: Distance transform to get blob centers
# ----------------------------------
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

# Unknown region (border between blobs)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# ----------------------------------
# Step 4: Marker labeling
# ----------------------------------
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1  # Ensure background is not 0 but 1
markers[unknown == 255] = 0

# ----------------------------------
# Step 5: Apply Watershed
# ----------------------------------
watershed_img = img.copy()
markers = cv2.watershed(watershed_img, markers)
watershed_img[markers == -1] = [0, 0, 255]  # Mark boundaries in red

# ----------------------------------
# Step 6: Display and Save the Result
# ----------------------------------
plt.figure(figsize=(14, 6))
plt.subplot(1, 3, 1)
plt.imshow(binary, cmap='gray')
plt.title("Binary Blobs")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(dist_transform, cmap='jet')
plt.title("Distance Transform")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(watershed_img, cv2.COLOR_BGR2RGB))
plt.title("Watershed Segmented")
plt.axis("off")

plt.tight_layout()
plt.show()

# Save output
output_path = "D:\Salman\Semesters\Sem7\Image Processing Lab\Lab Cycle 3\Question9\Watershed_Segmentation.jpg"
cv2.imwrite(output_path, watershed_img)
print(f"✅ Segmented image saved at: {output_path}")
