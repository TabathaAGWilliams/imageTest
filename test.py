import cv2
import numpy as np
from skimage.morphology import skeletonize

image = cv2.imread("images/whiteboard.png")

if image is None:
    print("Could not load image.")
    exit()


# Resize for easier viewing (optional)
image = cv2.resize(image, (1000, 700))

# Sobel in X direction
sobel_x = cv2.Sobel(
    image,           # source image
    cv2.CV_64F,    # output depth
    1,             # dx
    0,             # dy
    ksize=3        # kernel size
)

# Sobel in Y direction
sobel_y = cv2.Sobel(
    image,
    cv2.CV_64F,
    0,
    1,
    ksize=3
)

magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

# Convert to uint8 for display
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)
magnitude = cv2.convertScaleAbs(magnitude)

_, binary = cv2.threshold(magnitude, 127, 255, cv2.THRESH_BINARY)

# Convert to boolean image
binary_bool = binary > 0

# Skeletonize
skeleton = skeletonize(binary_bool)

# Convert back to uint8
skeleton_image = (skeleton * 255).astype(np.uint8)

cv2.imshow("Original", image)
cv2.imshow("Sobel x", sobel_x)
cv2.imshow("Sobel y", sobel_y)
cv2.imshow("Magnitude", magnitude)
cv2.imshow("Skeleton", skeleton_image)

cv2.waitKey(0)
cv2.destroyAllWindows()