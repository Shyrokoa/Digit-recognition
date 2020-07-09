import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical


class DigitRecognition:

    def __init__(self):
        self.X_train = ''
        self.X_test = ''
        self.y_train = ''
        self.y_test = ''
        self.model = ''

    def load_data(self):
        # download mnist data and split into train and test sets
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    def show_data(self, id):
        plt.imshow(self.X_test[id], cmap='Greys_r')

    '''
    By default, the shape of every image in the mnist dataset is 28 x 28, 
    so we will not need to check the shape of all the images. When using 
    real-world datasets, you may not be so lucky. 28 x 28 is also a 
    fairly small size, so the CNN will be able to run over each image 
    pretty quickly.
    '''

    def show_shape(self, id):
        print(self.X_train[id].shape)

    '''
    Next, we need to reshape our dataset inputs (X_train and X_test) to 
    the shape that our model expects when we train the model. The first 
    number is the number of images (60,000 for X_train and 10,000 for 
    X_test). Then comes the shape of each image (28x28). The last number 
    is 1, which signifies that the images are greyscale.
    '''

    def reshape_data(self):
        # reshape data to fit model
        self.X_train = self.X_train.reshape(60000, 28, 28, 1)
        self.X_test = self.X_test.reshape(10000, 28, 28, 1)

    '''
    We need to 'one-hot-encode' our target variable. This means that a 
    column will be created for each output category and a binary variable 
    is inputted for each category. For example, we saw that the first 
    image in the dataset is a 5. This means that the sixth number in our 
    array will have a 1 and the rest of the array will be filled with 0.
    '''

    def one_hot_encode(self):
        # one-hot encode target column
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)

    '''
    The model type that we will be using is Sequential. Sequential is the 
    easiest way to build a model in Keras. It allows you to build a model 
    layer by layer. We use the 'add()' function to add layers to our model. 
    Our first 2 layers are Conv2D layers. These are convolution layers that 
    will deal with our input images, which are seen as 2-dimensional matrices.
    64 in the first layer and 32 in the second layer are the number of nodes 
    in each layer. This number can be adjusted to be higher or lower, 
    depending on the size of the dataset. In our case, 64 and 32 work well, 
    so we will stick with this for now. Kernel size is the size of the filter 
    matrix for our convolution. So a kernel size of 3 means we will have a 
    3x3 filter matrix. Refer back to the introduction and the first image 
    for a refresher on this. Activation is the activation function for the 
    layer. The activation function we will be using for our first 2 layers 
    is the ReLU, or Rectified Linear Activation. This activation function 
    has been proven to work well in neural networks. Our first layer also 
    takes in an input shape. This is the shape of each input image, 28,28,1 
    as seen earlier on, with the 1 signifying that the images are greyscale. 
    In between the Conv2D layers and the dense layer, there is a 'Flatten' 
    layer. Flatten serves as a connection between the convolution and dense 
    layers. ‘Dense’ is the layer type we will use in for our output layer. 
    Dense is a standard layer type that is used in many cases for neural 
    networks. We will have 10 nodes in our output layer, one for each 
    possible outcome (0–9). The activation is 'softmax'. Softmax makes the 
    output sum up to 1 so the output can be interpreted as probabilities. 
    The model will then make its prediction based on which option has the 
    highest probability.
    '''

    def build_model(self):
        # create model
        self.model = Sequential()
        # add model layers
        self.model.add(Conv2D(256, kernel_size=5, activation='relu', input_shape=(28, 28, 1)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.1))
        self.model.add(Conv2D(256, kernel_size=5, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.1))
        self.model.add(Flatten())
        self.model.add(Dense(10, activation='softmax'))

    '''
    If you want to see the actual predictions that our model has made for 
    the test data, we can use the predict function. The predict function 
    will give an array with 10 numbers. These numbers are the probabilities 
    that the input image represents each digit (0–9). The array index with 
    the highest number represents the model prediction. The sum of each 
    array equals 1 (since each number is a probability). To show this, we 
    will show the predictions for the first 4 images in the test set. Note: 
    If we have new data, we can input our new data into the predict function 
    to see the predictions our model makes on the new data. Since we don’t 
    have any new unseen data, we will show predictions using the test set 
    for now.
    '''

    def check(folder, digit):
        img = cv2.imread(f'digit_predict_set/{folder}/{digit}.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dim = (28, 28)
        # resize image
        resized = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)
        resized = cv2.bitwise_not(resized)
        ret, resized = cv2.threshold(resized, 85, 255, cv2.THRESH_BINARY)
        plt.imshow(resized, cmap='Greys_r')
        a = resized
        a = np.expand_dims(a, axis=(0, 3))
        arr = self.model.predict(a)
        # print(arr[0])
        arr = arr.tolist()[0]
        print(f'Predicted number for {folder} is : {arr.index(max(arr))}')
