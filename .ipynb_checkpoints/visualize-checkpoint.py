import cv2
import matplotlib.pyplot as plt

# Load image and mask
image = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)

# Normalize mask if needed
mask = mask / 255.0

# Plot
plt.imshow(image, cmap='gray')
plt.imshow(mask, alpha=0.4, cmap='jet')
plt.title("Image + Mask Overlay")
plt.axis('off')
plt.show()