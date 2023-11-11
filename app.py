from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

le = LabelEncoder()
model = load_model('model1.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files:
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Read the uploaded image
            nparr = np.fromstring(uploaded_file.read(), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            real_image = cv2.resize(image, (500, 500))
            image = cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale

            _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

            # Apply Gaussian blur to reduce noise
            image = cv2.GaussianBlur(image, (5, 5), 0)

            # Use adaptive thresholding to handle varying lighting conditions
            binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            # Invert the binary image (make digits/symbols black on white)
            inverted_image = cv2.bitwise_not(binary_image)

            # Find contours in the inverted image
            contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create a copy of the original image for drawing bounding rectangles
            image_with_rectangles = image.copy()

            # Define a minimum contour area threshold to filter out small contours (adjust as needed)
            min_contour_area = 30
            max_contour_area = 500

            rectangle = []

            # Loop through the contours and draw bounding rectangles
            for contour in contours:
                if cv2.contourArea(contour) > min_contour_area and cv2.contourArea(contour) < max_contour_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    rectangle.append((x-1, y-4, w+2, h+8))
                    cv2.rectangle(image_with_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 2)

            y_total = 0

            for (x, y, w, h) in rectangle:
                y_total += y
            y_avg = y_total / len(rectangle)

            # Create a list to store valid bounding rectangles
            valid_rectangles = []

            for x, y, w, h in rectangle:
                if abs(y - y_avg) <= 100:
                    valid_rectangles.append((x, y, w, h))

            print(f'This is the valid bounding rectangles: {valid_rectangles}')

            # Delete all the photos in the crop_images folder
            path = 'crop_images'
            file_names = os.listdir(path)
            for file_ in file_names:
                image_path = os.path.join(path, file_)
                os.remove(image_path)
            
            valid_rectangles = sorted(valid_rectangles, key=lambda x: x[0])

            for i, (x, y, w, h) in enumerate(valid_rectangles):
                # Extract the portion of the image
                crop_image = real_image[y:y+h, x:x+w]

                # Check if the crop_image is not empty (i.e., it contains valid image data)
                if crop_image is not None and not crop_image.size == 0:
                    cv2.imwrite(f'crop_images/{i}.png', crop_image)
                else:
                    # Log an error and skip saving the image
                    print(f"Skipping image {i} because it's empty or invalid.")
            
            path = 'extracted_images'
            folders = []
            for folder in os.listdir(path):
                folders.append(folder)
            print(folders)

            labels = le.fit_transform(folders)

            model = load_model('model1.h5')
            path = 'crop_images'
            file_names = os.listdir(path)
            equation = []
            for file_  in file_names:
                image_path = os.path.join(path, file_)  # Corrected image path
                image = cv2.imread(image_path, 0)  # Load the image in grayscale
                if image is not None:  # Check if the image was loaded successfully
                    # Resize the image
                    image = cv2.resize(image, (24, 24))
                    image = image.reshape(1, 24, 24, 1)
                    image = image.astype('float32')/255  # Normalized pixel values to [0,1]
                    pred = model.predict(image)  # Corrected variable name
                    pred = np.argmax(pred)
                    equation.append(le.inverse_transform([pred])[0])

                    


                    print(f"This is the {le.inverse_transform([pred])} symbol")
        

                else:
                    print(f"Could not read image: {image_path}")
            

            


            
            # convert the list to string
            equation = ''.join(equation)

            return render_template('result.html', result=equation)

    return "No file uploaded"

if __name__ == '__main__':
    app.run(debug=True)
