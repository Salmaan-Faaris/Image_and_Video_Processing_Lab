import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Function to add salt-and-pepper noise
def add_salt_pepper_noise(image, salt_prob, pepper_prob):
    noisy_img = np.copy(image)
    total_pixels = image.size

    # Salt noise
    num_salt = np.ceil(salt_prob * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    noisy_img[tuple(coords)] = 255

    # Pepper noise
    num_pepper = np.ceil(pepper_prob * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    noisy_img[tuple(coords)] = 0

    return noisy_img

def main(image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not read the image.")
        return

    # salt-and-pepper noise
    noisy_image = add_salt_pepper_noise(image, 0.02, 0.02)

    # median filtering
    filtered_image = cv2.medianBlur(noisy_image, 3)

    titles = ['Original Image', 'Noisy Image', 'Median Filtered']
    images = [image, noisy_image, filtered_image]

    plt.figure(figsize=(10, 5))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "median_filter_comparison.png"))
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python median_filter.py <image_path> <output_dir>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
