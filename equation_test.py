import numpy as np
import cv2
import os
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

path = 'extracted_images'
folders = []
for folder in os.listdir(path):
    folders.append(folder)
print(folders)

labels = le.fit_transform(folders)


# Load the model
model = load_model('model1.h5')
path = 'crop_images'
file_names = os.listdir(path)
for file_ in file_names:
    image_path = os.path.join(path, file_)  # Corrected image path
    image = cv2.imread(image_path, 0)  # Load the image in grayscale
    if image is not None:  # Check if the image was loaded successfully
        # Resize the image
        image = cv2.resize(image, (24, 24))
        image = image.reshape(1, 24, 24, 1)
        image = image.astype('float32')/255  # Normalized pixel values to [0,1]
        pred = model.predict(image)  # Corrected variable name
        pred = np.argmax(pred)
        

        print(f"This is the {le.inverse_transform([pred])} symbol")
        

    else:
        print(f"Could not read image: {image_path}")
