import cv2
import keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist

from segmentatoin import getpictures
from tensorflow.keras.models import load_model




pictures = getpictures('test2.jpg')
model = load_model('./mnist_digit_recognition_model.h5')
strs = ''
# str = ['1.jpg','2.jpg','3.jpg','4.jpg','5.jpg','6.jpg','7.jpg','8.jpg','9.jpg','10.jpg','11.jpg']

for pic in pictures:
    print(pic)
        # str +=1

    # Load MNIST dataset
    # (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    #
    # # Build and train a simple CNN model
    # model = Sequential([
    #     Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)),
    #     MaxPooling2D((2, 2)),
    #     Flatten(),
    #     Dense(128, activation='relu'),
    #     Dense(10, activation='softmax')
    # ])
    #
    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # train_images = np.expand_dims(train_images, axis=-1)
    # test_images = np.expand_dims(test_images, axis=-1)

    # model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

    # Now, assuming you have an image 'input_image' with digits you want to detect
    input_image = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
    # resized_image = cv2.resize(input_image, (150, 150))
    # normalized_image = resized_image / 255.0
    input_data = np.expand_dims(input_image, axis=0)
    input_data = np.expand_dims(input_data, axis=-1)

    # Make predictions
    predictions = model.predict(input_data)
    detected_numbers = np.argmax(predictions, axis=1)
    strWala = str(detected_numbers[0])
    strs += strWala
    # print("Detected numbers:", detected_numbers[0])

print(strs)