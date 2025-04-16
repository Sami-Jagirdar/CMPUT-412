import cv2
import numpy as np

def nothing(x):
    pass

# Load image
image_path = 'lane3.png'
image = cv2.imread(image_path)
if image is None:
    print(f"Could not read image: {image_path}")
    exit()

# Create window with trackbars
cv2.namedWindow('HSV Threshold')
cv2.createTrackbar('H Min', 'HSV Threshold', 0, 179, nothing)
cv2.createTrackbar('H Max', 'HSV Threshold', 179, 179, nothing)
cv2.createTrackbar('S Min', 'HSV Threshold', 0, 255, nothing)
cv2.createTrackbar('S Max', 'HSV Threshold', 255, 255, nothing)
cv2.createTrackbar('V Min', 'HSV Threshold', 0, 255, nothing)
cv2.createTrackbar('V Max', 'HSV Threshold', 255, 255, nothing)

# Convert to HSV once
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

while True:
    # Get current trackbar positions
    h_min = cv2.getTrackbarPos('H Min', 'HSV Threshold')
    h_max = cv2.getTrackbarPos('H Max', 'HSV Threshold')
    s_min = cv2.getTrackbarPos('S Min', 'HSV Threshold')
    s_max = cv2.getTrackbarPos('S Max', 'HSV Threshold')
    v_min = cv2.getTrackbarPos('V Min', 'HSV Threshold')
    v_max = cv2.getTrackbarPos('V Max', 'HSV Threshold')
    
    # Create mask and apply it
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)
    
    # Show results
    cv2.imshow('Original', image)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)
    
    # Print current values
    print(f"Current HSV Range: [{h_min}, {s_min}, {v_min}] to [{h_max}, {s_max}, {v_max}]")
    
    # Break loop on ESC key
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # ESC key
        break

cv2.destroyAllWindows()