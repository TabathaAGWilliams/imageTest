import cv2
import numpy as np

# -----------------------------------
# Load image
# -----------------------------------
image = cv2.imread("whiteboard.jpg")

if image is None:
    print("Could not load image.")
    exit()

# Resize for easier viewing (optional)
image = cv2.resize(image, (1000, 700))

# -----------------------------------
# Convert to HSV
# -----------------------------------
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# -----------------------------------
# Red color ranges in HSV
# Red wraps around HSV color wheel,
# so we use TWO ranges
# -----------------------------------

lower_red1 = np.array([0, 100, 80])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([160, 100, 80])
upper_red2 = np.array([180, 255, 255])

# Create masks
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

# Combine masks
mask = mask1 + mask2

# -----------------------------------
# Noise cleanup
# -----------------------------------

kernel = np.ones((5, 5), np.uint8)

# Remove small noise
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Fill small gaps
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# -----------------------------------
# Find contours
# -----------------------------------

contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# -----------------------------------
# Draw only red outlines
# -----------------------------------

output = np.zeros_like(image)

for contour in contours:

    # Ignore tiny contours
    area = cv2.contourArea(contour)

    if area < 200:
        continue

    # Draw contour in red
    cv2.drawContours(
        output,
        [contour],
        -1,
        (0, 0, 255),   # Red in BGR
        3
    )

# -----------------------------------
# Show results
# -----------------------------------

cv2.imshow("Original", image)
cv2.imshow("Red Shapes Only", output)
cv2.imshow("Mask", mask)

cv2.waitKey(0)
cv2.destroyAllWindows()