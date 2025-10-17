import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Load the image
# -------------------------------
image_path = "D:\Salman\Semesters\Sem7\Image Processing Lab\Lab Cycle 3\Question4\Frozen Rose.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img = cv2.GaussianBlur(img, (5, 5), 0)

# -------------------------------
# Split and Merge Functions
# -------------------------------
def split_region(img, x, y, w, h, std_threshold, min_size):
    """
    Recursively split the region if variance > threshold and region > min_size
    """
    region = img[y:y+h, x:x+w]
    if region.size == 0:
        return []
    stddev = np.std(region)
    if stddev < std_threshold or w <= min_size or h <= min_size:
        mean_val = np.mean(region)
        return [(x, y, w, h, mean_val)]
    else:
        half_w, half_h = w // 2, h // 2
        regions = []
        regions += split_region(img, x, y, half_w, half_h, std_threshold, min_size)
        regions += split_region(img, x + half_w, y, half_w, half_h, std_threshold, min_size)
        regions += split_region(img, x, y + half_h, half_w, half_h, std_threshold, min_size)
        regions += split_region(img, x + half_w, y + half_h, half_w, half_h, std_threshold, min_size)
        return regions

def merge_regions(seg_img, regions, merge_threshold):
    """
    Merge neighboring regions if mean intensity difference < threshold
    """
    merged = seg_img.copy()
    for i, r1 in enumerate(regions):
        for j, r2 in enumerate(regions):
            if i != j:
                (x1, y1, w1, h1, m1) = r1
                (x2, y2, w2, h2, m2) = r2
                if abs(m1 - m2) < merge_threshold:
                    if abs(x1 - x2) <= max(w1, w2) and abs(y1 - y2) <= max(h1, h2):
                        merged[y1:y1+h1, x1:x1+w1] = (m1 + m2) / 2
                        merged[y2:y2+h2, x2:x2+w2] = (m1 + m2) / 2
    return merged

# -------------------------------
# Perform Split and Merge
# -------------------------------
def split_and_merge_segmentation(img, std_threshold=15, merge_threshold=10, min_size=16):
    h, w = img.shape
    regions = split_region(img, 0, 0, w, h, std_threshold, min_size)
    seg_img = np.zeros_like(img, dtype=np.float32)
    
    for (x, y, rw, rh, mean_val) in regions:
        seg_img[y:y+rh, x:x+rw] = mean_val
    
    merged_img = merge_regions(seg_img, regions, merge_threshold)
    return merged_img

# -------------------------------
# Try different min region sizes
# -------------------------------
sizes = [32, 16, 8]
plt.figure(figsize=(15, 5))

for i, size in enumerate(sizes):
    segmented = split_and_merge_segmentation(img, std_threshold=15, merge_threshold=10, min_size=size)
    plt.subplot(1, len(sizes), i+1)
    plt.imshow(segmented, cmap='gray')
    plt.title(f"Min Region = {size}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# -------------------------------
# Save one of the outputs
# -------------------------------
output_path = "D:\Salman\Semesters\Sem7\Image Processing Lab\Lab Cycle 3\Question8\SplitMergeSegmentation.jpg"
cv2.imwrite(output_path, segmented)
print(f"✅ Segmented image saved at: {output_path}")
