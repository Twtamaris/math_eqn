import cv2
import numpy as np

# Load the image
image = cv2.imread('1.jpg', 0)  # Load the image in grayscale
image = cv2.resize(image, (500, 500))

# Threshold the image to create a binary image
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
print(binary_image)

# Invert the binary image (make digits/symbols black on white)
inverted_image = cv2.bitwise_not(binary_image)

# Find contours in the inverted image
contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image for drawing bounding rectangles
image_with_rectangles = image.copy()

# Define a minimum contour area threshold to filter out small contours (adjust as needed)
min_contour_area = 0 # Adjust this threshold

# Loop through the contours and draw bounding rectangles
for contour in contours:
    
    if cv2.contourArea(contour) > min_contour_area:
        x, y, w, h = cv2.boundingRect(contour)

        hehe = cv2.rectangle(image_with_rectangles, (x, y), (x + w+200, y + h+200), (0, 255, 0), 2)
        
        

# Merge overlapping bounding rectangles (your merge_overlapping_rectangles function)

# Define the merge_overlapping_rectangles function
def merge_overlapping_rectangles(rectangles):
    merged_rectangles = []
    while len(rectangles) > 0:
        x1, y1, w1, h1 = rectangles[0]
        x2, y2, w2, h2 = rectangles[0]
        for rect in rectangles[1:]:
            x2, y2, w2, h2 = rect
            if x2 < x1 + w1 and x2 + w2 > x1 and y2 < y1 + h1 and y2 + h2 > y1:
                x1 = min(x1, x2)
                y1 = min(y1, y2)
                w1 = max(x1 + w1, x2 + w2) - x1
                h1 = max(y1 + h1, y2 + h2) - y1
                rectangles.remove(rect)
        merged_rectangles.append((x1, y1, w1, h1))
        rectangles.remove(rectangles[0])
    return merged_rectangles

# Extract bounding rectangles from contours
bounding_rectangles = [cv2.boundingRect(contour) for contour in contours]

# Merge overlapping bounding rectangles
merged_rectangles = merge_overlapping_rectangles(bounding_rectangles)


# Extract bounding rectangles from contours
bounding_rectangles = [cv2.boundingRect(contour) for contour in contours]

# Merge overlapping bounding rectangles
merged_rectangles = merge_overlapping_rectangles(bounding_rectangles)

# Create a copy of the original image for drawing merged bounding rectangles
image_with_merged_rectangles = image.copy()

# Draw the merged bounding rectangles
for x, y, w, h in merged_rectangles:
    cv2.rectangle(image_with_merged_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the images
# cv2.imshow("Original Image", image)
# cv2.imshow("Inverted Image with Contours", image_with_rectangles)
# cv2.imshow("Inverted Image with Merged Rectangles", image_with_merged_rectangles)
print("This is countour", contours)
# cv2.imshow("Binary_image", inverted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
