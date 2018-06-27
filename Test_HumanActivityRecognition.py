import os

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout, Flatten
from keras.models import model_from_json
from sklearn.utils import class_weight
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
#from keras import backend as K
from keras import optimizers

from sklearn.metrics import confusion_matrix
from humanactivity import classifiers
from data_loader import extract_acc_gyr_data_from_mat_file, extract_normalize_acc_gyr_data_from_mat_file, \
    extract_acc_gyr_bar_data_from_mat_file, segment_data, windows, \
    vis_acc_gyr_data, vis_acc_gyr_bar_data, unique_activity_classes

c_work_dir = os.getcwd()
model_dir = c_work_dir + '\\models\\'
dir_path_train = 'F:\\2018531_153334_PMEAS_RED_CAHRS_EKF_FS_MIMU_100_FS_BAR_16\\xpypzp\\'
train_files = os.listdir(dir_path_train)

dir_path_test = 'F:\\2018531_153334_PMEAS_RED_CAHRS_EKF_FS_MIMU_100_FS_BAR_16\\xnynzp\\'
test_files = os.listdir(dir_path_test)

dir_path = 'F:\\2018529_164234_PMEAS_RED_CAHRS_EKF_FS_MIMU_40_FS_BAR_20'
mat_filename = '\\xpypzp_DP13_011.mat'

X_train_acc_gyr, y_train, y_train_unit8, class_weights = extract_acc_gyr_data_from_mat_file(dir_path, mat_filename)
vis_acc_gyr_data(X_train_acc_gyr)

X_train_acc_gyr_bar, y_train2, y_train_unit8_2, class_weights = extract_acc_gyr_bar_data_from_mat_file(dir_path, mat_filename)
vis_acc_gyr_bar_data(X_train_acc_gyr_bar)

json_model = []
json_weight = []
yaml_model = []
yaml_weight = []

model_files = os.listdir(model_dir)
for f in model_files:
    if f == "model_json.json":
        json_model = model_dir + f
    elif f == "model_json.h5":
        json_weight = model_dir + f
    elif f == "model_yaml.yaml":
        yaml_model = model_dir + f
    elif f == "model_yaml.h5":
        yaml_weight = model_dir + f

# start model training and testing here
if (json_model != []) & (json_weight != []):
    loaded_json_model = classifiers.load_json_model_and_weights(json_model, json_weight)
    model = loaded_json_model

    print('loading json model')
elif (yaml_model != []) & (yaml_weight != []):
    print('yaml model')
else:
    print('Start training model on data')
    num_input_channels = 6
    num_input_units = 8
    num_classes = 6

    epoch_num = 20
    batch_num = 50

    model = classifiers.build_neural_network_model(
        num_input_channels, num_input_units, num_classes)

    for train_file in train_files:
        print("file: "+train_file)

        if train_file.find('DP13_047') > 0:
            print('Ignoring ' + train_file)
        else:
            # x_train_acc_gyr, y_train, y_train_unit8, class_weights = \
            #     extract_acc_gyr_data_from_mat_file(dir_path, "\\"+train_file)

            x_train_acc_gyr, y_train, y_train_unit8, class_weights = \
                extract_normalize_acc_gyr_data_from_mat_file(dir_path, "\\"+train_file)

###

            segments, labels = segment_data(dir_path, "\\"+train_file, 100)
            labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)
            reshaped_segments = segments.reshape(len(segments), 1, 100, 6)

            train_test_split = np.random.rand(len(reshaped_segments)) < 0.70
            train_x = reshaped_segments[train_test_split]
            train_y = labels[train_test_split]
            test_x = reshaped_segments[~train_test_split]
            test_y = labels[~train_test_split]

            numOfRows = segments.shape[1]
            numOfColumns = segments.shape[2]
            numChannels = 1
            numFilters = 128  # number of filters in Conv2D layer
            # kernal size of the Conv2D layer
            kernalSize1 = 2
            # max pooling window size
            poolingWindowSz = 2
            # number of filters in fully connected layers
            numNueronsFCL1 = 128
            numNueronsFCL2 = 128
            # split ratio for test and validation
            trainSplitRatio = 0.8
            # number of epochs
            Epochs = 10
            # batchsize
            batchSize = 10
            # number of total clases
            numClasses = labels.shape[1]
            # dropout ratio for dropout layer
            dropOutRatio = 0.2
            # reshaping the data for network input
            reshapedSegments = segments.reshape(segments.shape[0], numOfRows, numOfColumns, 1)
            # splitting in training and testing data
            trainSplit = np.random.rand(len(reshapedSegments)) < trainSplitRatio
            trainX = reshapedSegments[trainSplit]
            testX = reshapedSegments[~trainSplit]
            trainX = np.nan_to_num(trainX)
            testX = np.nan_to_num(testX)
            trainY = labels[trainSplit]
            testY = labels[~trainSplit]


            def cnnModel():
                model = Sequential()
                # adding the first convolutionial layer with 32 filters and 5 by 5 kernal size, using the rectifier as the activation function
                model.add(Conv2D(numFilters, (kernalSize1, kernalSize1), input_shape=(numOfRows, numOfColumns, 1),
                                 activation='relu'))
                # adding a maxpooling layer
                model.add(MaxPooling2D(pool_size=(poolingWindowSz, poolingWindowSz), padding='valid'))
                # adding a dropout layer for the regularization and avoiding over fitting
                model.add(Dropout(dropOutRatio))
                # flattening the output in order to apply the fully connected layer
                model.add(Flatten())
                # adding first fully connected layer with 256 outputs
                model.add(Dense(numNueronsFCL1, activation='relu'))
                # adding second fully connected layer 128 outputs
                model.add(Dense(numNueronsFCL2, activation='relu'))
                # adding softmax layer for the classification
                model.add(Dense(numClasses, activation='softmax'))
                # Compiling the model to generate a model
                adam = optimizers.Adam(lr=0.001, decay=1e-6)
                model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
                return model


            model = cnnModel()
            for layer in model.layers:
                print(layer.name)
            model.fit(trainX, trainY, validation_split=1 - trainSplitRatio, epochs=10, batch_size=batchSize, verbose=2)
            score = model.evaluate(testX, testY, verbose=2)
            print('Baseline Error: %.2f%%' % (100 - score[1] * 100))
###

            # unbalanced class distribution
            #classifiers.fit_neural_network_model(model, x_train_acc_gyr, y_train_unit8, epoch_num, batch_num)

            classifiers.fit_balanced_neural_network_model(
                model, x_train_acc_gyr, y_train_unit8, class_weights, epoch_num, batch_num)

    classifiers.save_model_to_json(model)
    classifiers.save_model_to_yaml(model)
    ###################################################


print('Commence evaluation')
# predict on all data
for file in test_files:
    print("file: " + file)

    x_test_acc_gyr, y_test, y_test_unit8, c_weights = extract_acc_gyr_data_from_mat_file(dir_path_test, file)

    y_predict = model.predict(x_test_acc_gyr)
    y_predict_classes = model.predict_classes(x_test_acc_gyr)
    y_val = y_predict.argmax(axis=1)

    conf_mat = confusion_matrix(y_test, y_predict_classes)

    plt.figure()
    plt.plot(y_predict_classes)
    plt.plot(y_val)
    plt.plot(y_test)
print('Finished')
