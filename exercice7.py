#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import numpy as np


#####################################################################################################################
#####################################################################################################################
def ANN_model():

    model = Sequential()

    model.add(Dense(4, input_dim=2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])

    return model


#####################################################################################################################
#####################################################################################################################
def trainAndPredictANN(X_train, Y_train, X_test, Y_test):

    # Application 2 - Step 6 - Call the cnn_model function
    model = ANN_model()

    # TODO - Application 2 - Step 8 - Train the model
    model.fit(X_train, Y_train, epochs=5000, verbose=0)

    # TODO - Application 2 - Step 8 - System evaluation - compute and display the prediction error
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("CNN Error: {:.2f}".format(100 - scores[1] * 100))
    print("accuracy = " + str(scores[1] * 100))

#####################################################################################################################
#####################################################################################################################
def main():
    # Input data
    P = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    # Labels
    # for AND
    # t = [0, 0, 0, 1]

    # for OR
    t = [0, 1, 1, 1]

    trainAndPredictANN(P, t, P, t)

#####################################################################################################################
#####################################################################################################################
if __name__ == '__main__':
    main()
#####################################################################################################################
#####################################################################################################################
