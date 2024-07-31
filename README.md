# direction-prediction-neural-network

A neural network that predicts the direction of an arrow, takes input a 64 x 64 pixels vector image and outputs the direction label.

The traning and validation dataset is available in the root folder named `dataset`.

It contains **16 tranining images** and **4 validation/test images**.

## Requirements
```py
pip install tensorflow keras PIL numpy
```

## Building the model
```py
python3 model.py
```

The Model achives a maximum test accuracy of 0.5 on test dataset, I don't know why, may be because there are only 4 images for testing model accuracy. But If you train the model yourself you'll see the accuracy to reach **1.0** after just 7 or 8 epochs, which is great.

## Testing the model
```py
python3 predict.py
```

You can even draw your own images in **color** 4 channels, on a 64 by 64 pixel canvas and change `prediction.py` with your image's path to test the model.


I'm just starting out with Machine Learning, and this is the first model I created by myself after learning these concepts:
- Neuron
- Activation Functions (ReLU, Tanh, Sigmoid, Softmax)
- Dense Layer
- Convolutional Layer
- MaxPooling2D Layer
- Flatten Layer
- Optimizer
- Mean Square Error
- Spare Categorial Cross Entropy Loss
- and a concrete implementation from [TheIndependentCode's Neural Network Implementation](https://github.com/TheIndependentCode/Neural-Network).

I learned these concepts from ChatGPT and from other sources like youtube and blog posts.