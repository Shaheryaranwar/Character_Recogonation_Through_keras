import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained digit recognition model
# model = load_model('path_to_your_trained_model.h5')  # Replace with the path to your model file

# Load the input image

def getpictures(fileName):

    fileNames = []
    input_image = cv2.imread(fileName)
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding or other preprocessing techniques to isolate digits
    # Example: Using adaptive thresholding
    _, thresh = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours to detect digits
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    detected_numbers = []
    count = 0
    # Iterate through contours (presumed to be digits) and recognize each digit
    for i,contour in enumerate(contours):
        count= count+1
        x, y, w, h = cv2.boundingRect(contour)

        # Extract individual digit
        digit = thresh[y:y + h, x:x + w]

        # Resize digit to 28x28 (similar to MNIST images)
        resized_digit = cv2.resize(digit, (28, 28))

        # Normalize pixel values
        normalized_digit = resized_digit / 255.0

        # Reshape for model input (add batch and channel dimensions)
        input_data = normalized_digit.reshape(1, 28, 28, 1)
        newFile = './'+str(count)+".jpg"
        cv2.imwrite(newFile, resized_digit.astype('uint8'))
        fileNames.append(newFile)
        # Make prediction using the model
        # prediction = np.argmax(model.predict(input_data), axis=1)
        # detected_numbers.append(prediction[0])
    return fileNames
    # Output detected numbers
    # print("Detected numbers:", detected_numbers)
