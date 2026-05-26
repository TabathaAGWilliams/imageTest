import cv2
import numpy as np


image = cv2.imread("images/whiteboard.png")

if image is None:
    print("Could not load image.")
    exit()


# Resize for easier viewing (optional)
image = cv2.resize(image, (1000, 700))

cv2.imshow("Original", image)

cv2.waitKey(0)
cv2.destroyAllWindows()