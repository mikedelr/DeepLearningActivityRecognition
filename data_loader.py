import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from humanactivity import classifiers
from sklearn.utils import class_weight
from scipy import stats


unique_activity_classes = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint8)


def extract_acc_gyr_data_from_mat_file(path, file):
    mat_data = sio.loadmat(path + file)
    x_train = np.zeros([len(mat_data['accX']), 6], dtype=np.float64)
    x_train[:, 0] = np.array(mat_data['accX']).ravel()
    x_train[:, 1] = np.array(mat_data['accY']).ravel()
    x_train[:, 2] = np.array(mat_data['accZ']).ravel()
    x_train[:, 3] = np.array(mat_data['gyrX']).ravel()
    x_train[:, 4] = np.array(mat_data['gyrY']).ravel()
    x_train[:, 5] = np.array(mat_data['gyrZ']).ravel()

    y_train = np.array(mat_data['act']).ravel()
    y_train_unit8 = classifiers.get_dummies_from_map(y_train, unique_activity_classes)
    y_train_unit8 = np.array(y_train_unit8)

    # class_weights = class_weight.compute_class_weight('balanced', np.unique(unique_activity_classes), y_train)
    freqw = len(y_train) / (len(unique_activity_classes) * np.bincount(y_train).astype(np.float64))
    class_weights = freqw[1:7]
    return x_train, y_train, y_train_unit8, class_weights


def extract_normalize_acc_gyr_data_from_mat_file(path, file):
    mat_data = sio.loadmat(path + file)
    x_train = np.zeros([len(mat_data['accX']), 6], dtype=np.float64)
    x_train[:, 0] = feature_normalize(np.array(mat_data['accX']).ravel())
    x_train[:, 1] = feature_normalize(np.array(mat_data['accY']).ravel())
    x_train[:, 2] = feature_normalize(np.array(mat_data['accZ']).ravel())
    x_train[:, 3] = feature_normalize(np.array(mat_data['gyrX']).ravel())
    x_train[:, 4] = feature_normalize(np.array(mat_data['gyrY']).ravel())
    x_train[:, 5] = feature_normalize(np.array(mat_data['gyrZ']).ravel())

    y_train = np.array(mat_data['act']).ravel()
    y_train_unit8 = classifiers.get_dummies_from_map(y_train, unique_activity_classes)
    y_train_unit8 = np.array(y_train_unit8)

    # class_weights = class_weight.compute_class_weight('balanced', np.unique(unique_activity_classes), y_train)
    freqw = len(y_train) / (len(unique_activity_classes) * np.bincount(y_train).astype(np.float64))
    class_weights = freqw[1:7]
    return x_train, y_train, y_train_unit8, class_weights


def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu)/sigma


def windows(data, size):
    start = 0
    while start < len(data):
        yield int(start), int(start + size)
        # 50% overlap
        start += (size / 2)


def segment_data(path, file, window_size=100):
    mat_data = sio.loadmat(path + file)
    x_train = np.zeros([len(mat_data['accX']), 6], dtype=np.float64)
    x_train[:, 0] = feature_normalize(np.array(mat_data['accX']).ravel())
    x_train[:, 1] = feature_normalize(np.array(mat_data['accY']).ravel())
    x_train[:, 2] = feature_normalize(np.array(mat_data['accZ']).ravel())
    x_train[:, 3] = feature_normalize(np.array(mat_data['gyrX']).ravel())
    x_train[:, 4] = feature_normalize(np.array(mat_data['gyrY']).ravel())
    x_train[:, 5] = feature_normalize(np.array(mat_data['gyrZ']).ravel())

    y_train = np.array(mat_data['act']).ravel()

    segments = np.empty((0, window_size, 6))
    labels = np.empty((0))

    for (start, end) in windows(x_train[:, 0], window_size):
        ax = x_train[start:end, 0]
        ay = x_train[start:end, 1]
        az = x_train[start:end, 2]
        wx = x_train[start:end, 3]
        wy = x_train[start:end, 4]
        wz = x_train[start:end, 5]
        if len(x_train[start:end, 0]) == window_size:
            segments = np.vstack([segments, np.dstack([ax, ay, az, wx, wy, wz])])
            labels = np.append(labels, stats.mode(y_train[start:end])[0][0])
    return segments, labels


def vis_acc_gyr_data(x_train):
    # matplotlib inline
    fig = plt.figure()
    plt.interactive(True)
    hsub1 = plt.subplot(2, 1, 1)
    h_ax, = plt.plot(x_train[:, 0], label='a_{x}')
    h_ay, = plt.plot(x_train[:, 1], label='a_{y}')
    h_az, = plt.plot(x_train[:, 2], label='a_{z}')
    plt.legend(handles=[h_ax, h_ay, h_az], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=3, mode="none", borderaxespad=0.)
    plt.grid()
    plt.show()
    plt.subplot(2, 1, 2, sharex=hsub1)
    h_wx, = plt.plot(x_train[:, 3], label='w_{x}')
    h_wy, = plt.plot(x_train[:, 4], label='w_{y}')
    h_wz, = plt.plot(x_train[:, 5], label='w_{z}')
    plt.legend(handles=[h_wx, h_wy, h_wz], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=3, mode="none", borderaxespad=0.)
    plt.grid()
    plt.show()


def extract_acc_gyr_bar_data_from_mat_file(path, file):
    mat_data = sio.loadmat(path + file)
    x_train = np.zeros([len(mat_data['accX']), 7], dtype=np.float64)
    x_train[:, 0] = np.array(mat_data['accX']).ravel()
    x_train[:, 1] = np.array(mat_data['accY']).ravel()
    x_train[:, 2] = np.array(mat_data['accZ']).ravel()
    x_train[:, 3] = np.array(mat_data['gyrX']).ravel()
    x_train[:, 4] = np.array(mat_data['gyrY']).ravel()
    x_train[:, 5] = np.array(mat_data['gyrZ']).ravel()
    x_train[:, 6] = np.array(mat_data['pre']).ravel()

    y_train = np.array(mat_data['act']).ravel()
    y_train_unit8 = classifiers.get_dummies_from_map(y_train, unique_activity_classes)
    y_train_unit8 = np.array(y_train_unit8)

    class_weights = class_weight.compute_class_weight('balanced', np.unique(unique_activity_classes), y_train)

    return x_train, y_train, y_train_unit8, class_weights


def vis_acc_gyr_bar_data(x_train):
    # matplotlib inline
    fig = plt.figure()
    plt.interactive(True)
    hsub1 = plt.subplot(3, 1, 1)
    h_ax, = plt.plot(x_train[:, 0], label='a_{x}')
    h_ay, = plt.plot(x_train[:, 1], label='a_{y}')
    h_az, = plt.plot(x_train[:, 2], label='a_{z}')
    plt.legend(handles=[h_ax, h_ay, h_az], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=3, mode="none", borderaxespad=0.)
    plt.grid()
    plt.show()
    plt.subplot(3, 1, 2, sharex=hsub1)
    h_wx, = plt.plot(x_train[:, 3], label='w_{x}')
    h_wy, = plt.plot(x_train[:, 4], label='w_{y}')
    h_wz, = plt.plot(x_train[:, 5], label='w_{z}')
    plt.legend(handles=[h_wx, h_wy, h_wz], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=3, mode="none", borderaxespad=0.)
    plt.grid()
    plt.show()

    plt.subplot(3, 1, 3, sharex=hsub1)
    h_pre, = plt.plot(x_train[:, 6], label='p')
    plt.legend(handles=[h_pre])
    plt.grid()
    plt.show()



