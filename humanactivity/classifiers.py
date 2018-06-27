import os

import numpy as np

import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import time


from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout, Flatten
from keras.models import model_from_json
from sklearn.utils import class_weight
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
#from keras import backend as K
from keras import optimizers


def load_json_model_and_weights(json_model, json_weight):
    # load json and create model
    json_file = open(json_model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(json_weight)
    return loaded_model


def neural_network_model(x_train, y_train):
    feat_dims = x_train.shape
    parameters = feat_dims[1]

    AccuracyArr = np.empty(0)
    sArr = np.empty(0)
    for s in 6, 8, 10, 12, 14, 16, 18:
        print('s: ' + str(s))
        # Compile model
        model = Sequential()
        model.add(Dense(s, input_dim=parameters, activation='relu'))
        model.add(Dense(8, activation='relu'))

        # the output layer must have one neuron per class to be identified
        model.add(Dense(6, activation='sigmoid'))
        # Compile model
        # model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy', 'categorical_accuracy'])
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_accuracy'])
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy', 'categorical_accuracy'])
        print(model.summary())
        # one epoch = one forward pass and one backward pass of all the training examples
        # batch size = the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.
        # number of iterations = number of passes, each pass using [batch size] number of examples. To be clear, one pass = one forward pass + one backward pass (we do not count the forward pass and backward pass as two different passes).
        model.fit(x_train, y_train, epochs=20, batch_size=100)
        # Final evaluation of the model
        scores = model.evaluate(x_train, y_train, verbose=0)
        AccuracyArr = np.append(AccuracyArr, (scores[1] * 100))
        sArr = np.append(sArr, s)
        print("Accuracy: %.2f%%" % (scores[1] * 100))


def build_neural_network_model(num_input_channels, num_input_units, num_classes):
    model = Sequential()
    model.add(Dense(num_input_units, input_dim=num_input_channels, activation='relu'))
    model.add(Dense(8, activation='relu'))

    # the output layer must have one neuron per class to be identified
    model.add(Dense(num_classes, activation='sigmoid'))
    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def fit_neural_network_model(model, x_train, y_train, epoch_num, batch_num):
    model.fit(x_train, y_train, epochs=epoch_num, batch_size=batch_num)


def fit_balanced_neural_network_model(model, x_train, y_train, class_weights, epoch_num, batch_num):
    model.fit(x_train, y_train, epochs=epoch_num, batch_size=batch_num, class_weight=class_weights)


def get_dummies_from_map(vector, map):
    n_dims = vector.shape
    n_item = map.shape
    dummies = np.zeros([n_dims[0], n_item[0]], dtype=np.uint8)
    for i in map:
        dummies[vector == i, i - 1] = 1
    return dummies


def model_predict(model, x_test):
    model.predict(x_test)


def model_evaluate(model, x_test, y_test):
    model.evaluate(x_test, y_test)


def save_model_to_json(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model_json.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_json.h5")
    print("Saved model to disk")


def save_model_to_yaml(model):
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open("model_yaml.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights("model_yaml.h5")
    print("Saved model to disk")

def cnnModel():

    model = Sequential()
    # adding the first convolutionial layer with 32 filters and 5 by 5 kernal size, using the rectifier as the activation function
    model.add(Conv2D(numFilters, (kernalSize1, kernalSize1),
                     input_shape=(numOfRows, numOfColumns, 1), activation='relu'))

    # adding a maxpooling layer
    model.add(MaxPooling2D(pool_size=(poolingWindowSz, poolingWindowSz), padding='valid'))
    # adding a dropout layer for the regularization and avoiding over fitting
    model.add(Dropout(dropOutRatio))
    # flattening the output in order to apply the fully connected layer
    model.add(Flatten())
    # adding first fully connected layer with 256 outputs
    model.add(Dense(numNueronsFCL1, activation='relu'))
    #adding second fully connected layer 128 outputs
    model.add(Dense(numNueronsFCL2, activation='relu'))
    # adding softmax layer for the classification
    model.add(Dense(numClasses, activation='softmax'))
    # Compiling the model to generate a model
    adam = optimizers.Adam(lr=0.001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model
###################################################
