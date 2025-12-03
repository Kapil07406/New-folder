import cv2

# Load the image
image = cv2.imread('boy.jpg')

# Create a resizable window
cv2.namedWindow('Loaded Image', cv2.WINDOW_NORMAL)

# Set a specific window size (does not resize the image)
cv2.resizeWindow('Loaded Image', 800, 500)

# Display the image
cv2.imshow('Loaded Image', image)
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close the window

# Print image properties (Height, Width, Channels)
print(f"Image Dimensions: {image.shape}")