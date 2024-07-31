import os
import numpy as np
from tensorflow.keras import layers, models
from PIL import Image

def get_label(arrow):
    if arrow == 'arrow-up':
        return 0
    if arrow == 'arrow-right':
        return 1
    if arrow == 'arrow-down':
        return 2
    return 3

def get_dataset(dataset):
    labels_dir = os.listdir(dataset)
    X = []
    Y = []
    for dir in labels_dir:
        files = os.listdir(f'{dataset}/{dir}')
        x = [Image.open(f'{dataset}/{dir}/{file}') for file in files]
        y = [get_label(dir) for file in files]
        for image in x:
            X.append(image)
        for label in y:
            Y.append(label)
    X = np.array(X)
    Y = np.array(Y)
    return (X, Y)


(train_x, train_y) = get_dataset("/home/omegaui/learning-ai/arrow-model/dataset/train")
(test_x, test_y) = get_dataset("/home/omegaui/learning-ai/arrow-model/dataset/validation")

train_x = train_x / 255.0
test_x = test_x / 255.0

model = models.Sequential([
    layers.Conv2D(64, (8, 8), activation='tanh', input_shape=(64, 64, 4)),
    layers.MaxPooling2D((4, 4)),
    layers.Conv2D(64, (8, 8), activation='sigmoid'),
    layers.MaxPooling2D((4, 4)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(4, activation='softmax'),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(train_x, train_y, epochs=1000, 
    validation_data=(test_x, test_y)
)

test_loss, test_acc = model.evaluate(test_x, test_y)
print(f'Test accuracy: {test_acc}')

model.save('arrow_model.keras')

