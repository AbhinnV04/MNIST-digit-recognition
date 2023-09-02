import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np  


def test(image_number):
    image_path = f'digits/digits_{image_number}.png'
    
    if not os.path.isfile(image_path):
        print(f"Image {image_path} does not exist.")
        return

    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read image {image_path}.")
            return

        img = img[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")

model = tf.keras.models.load_model('digit.model')

test(1)
test(2)
test(3)
test(4)
test(5)
