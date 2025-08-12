import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import filters, feature, color

# Function to apply different edge detection methods
def edge_detection_comparison(image_path):
    # Read and convert to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found. Check the path.")

    # 1) Sobel
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)

    # 2) Prewitt
    prewittx = ndi.convolve(img, np.array([[ -1, 0, 1],
                                           [ -1, 0, 1],
                                           [ -1, 0, 1]]))
    prewitty = ndi.convolve(img, np.array([[ 1,  1,  1],
                                           [ 0,  0,  0],
                                           [-1, -1, -1]]))
    prewitt = np.hypot(prewittx, prewitty)

    # 3) Roberts
    roberts_cross_v = np.array([[1, 0],
                                [0, -1]])
    roberts_cross_h = np.array([[0, 1],
                                [-1, 0]])
    roberts_v = ndi.convolve(img, roberts_cross_v)
    roberts_h = ndi.convolve(img, roberts_cross_h)
    roberts = np.hypot(roberts_v, roberts_h)

    # 4) Laplacian of Gaussian (LoG)
    log = cv2.GaussianBlur(img, (3, 3), 0)
    log = cv2.Laplacian(log, cv2.CV_64F)

    # 5) Canny
    canny = cv2.Canny(img, 100, 200)

    # Display montage
    titles = ['Original', 'Sobel', 'Prewitt', 'Roberts', 'LoG', 'Canny']
    images = [img, sobel, prewitt, roberts, log, canny]

    plt.figure(figsize=(12, 8))
    for i in range(len(images)):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("edge_detection_comparison.png")
    plt.show()

# Example usage
image_path = "D:\\Salman\\Semesters\\Sem4\\Digital Signal Processing\\Frozen Rose.jpg"
edge_detection_comparison(image_path)
