import numpy as np
import tensorflow as tf
from PIL import Image


# Load the saved model
loaded_model = tf.keras.models.load_model('/home/omegaui/learning-ai/arrow-model/arrow_model.keras')

img = Image.open('/home/omegaui/learning-ai/arrow-model/dataset/train/arrow-down/img-1.png')

img_array = np.array(img).reshape((1, 64, 64, 4))

img_array = img_array / 255.0

print(img_array.shape)

predictions = loaded_model.predict(img_array)

prediction = np.argmax(predictions)


def get_label(prediction):
    if prediction == 0:
        return 'arrow-up'
    if prediction == 1:
        return 'arrow-right'
    if prediction == 2:
        return 'arrow-down'
    return 'arrow-left'

print(get_label(prediction))

