import cv2
import numpy as np
import os

image_name = 'images/5.jpg'

# Load the image
real_image = cv2.imread(image_name) 
real_image = cv2.resize(real_image, (500, 500))
image = cv2.imread(image_name, 0)  # Load the image in grayscale
image1 = cv2.resize(image, (500, 500))

_, binary_image = cv2.threshold(image1, 128, 255, cv2.THRESH_BINARY)

# Apply Gaussian blur to reduce noise
image = cv2.GaussianBlur(image1, (5, 5), 0)

# Use adaptive thresholding to handle varying lighting conditions
binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Invert the binary image (make digits/symbols black on white)
inverted_image = cv2.bitwise_not(binary_image)

# Find contours in the inverted image
contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image for drawing bounding rectangles
image_with_rectangles = image.copy()

# Define a minimum contour area threshold to filter out small contours (adjust as needed)
min_contour_area = 30# Adjust this threshold
max_contour_area = 500

rectangle = []



# Loop through the contours and draw bounding rectangles
for contour in contours:
    if cv2.contourArea(contour) > min_contour_area and cv2.contourArea(contour) < max_contour_area:
        x, y, w, h = cv2.boundingRect(contour)
        rectangle.append((x-1, y-4, w+2, h+8))
        cv2.rectangle(image_with_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 2)

y_total = 0

for (x,y,w,h) in rectangle:
    y_total += y
y_avg = y_total/len(rectangle)


for (x,y,w,h) in rectangle:
    if abs(y-y_avg) > 100:
        rectangle.remove((x,y,w,h))

        




# Extract bounding rectangles from contours
# sort the rectangle on basis of x
bounding_rectangles = sorted(rectangle, key=lambda x: x[0])
print(bounding_rectangles)

# delete all the photos in crop_images folder
path = 'crop_images'
file_names = os.listdir(path)
for file_ in file_names:
    image_path = os.path.join(path, file_)  # Corrected image path
    os.remove(image_path)

for i,(x,y,w,h) in enumerate(bounding_rectangles):
    
    # Extract the portion of the image
    crop_image = real_image[y:y+h, x:x+w]
    cv2.imwrite(f'crop_images/{i}.png', crop_image)

# Draw bounding rectangles on a copy of the original image
image_with_bounding_rectangles = image1.copy()

for rect in bounding_rectangles:
    x, y, w, h = rect
    cv2.rectangle(image_with_bounding_rectangles, (x,y), (x+w,y+h), (0,255,0), 2)

cv2.imshow("Inverted Image with Bounding Rectangles", image_with_bounding_rectangles)
cv2.waitKey(0)
cv2.destroyAllWindows()
