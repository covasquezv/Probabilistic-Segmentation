import h5py
import cv2
import numpy as np
import math
import matplotlib.pylab as plt

def load(X, y, i, image_size=128):
    image = X; _mask_image = y

    image = cv2.resize(image, (image_size, image_size))
    mask = np.zeros((image_size, image_size, 1))

    _mask_image = cv2.resize(_mask_image, (image_size, image_size)) #128x128
    _mask_image = np.expand_dims(_mask_image, axis=-1)
    mask = np.maximum(mask, _mask_image)

    ## Normalizaing
    image = image/255.0
    mask = mask/255.0

    tmp_im = image.reshape(image.shape[0], image.shape[1])
    tmp_mask = mask.reshape(mask.shape[0], mask.shape[1])

    # fig = plt.figure(i+1)
    # plt.set_cmap('gray')
    # fig.add_subplot(121)
    # plt.imshow(tmp_im)
    # fig.add_subplot(122)
    # plt.imshow(tmp_mask)
    # # plt.imshow(s[i][:,:,0])
    # plt.savefig(str(i+1)+'_.png')

    return image, mask

def read_data(PATH):

    X_train = h5py.File(PATH+'X_train.hdf5', 'r')
    X_tr = X_train.get('X_train')[:]
    y_train = h5py.File(PATH+'y_train.hdf5', 'r')
    y_tr = y_train.get('y_train')[:]

    X_val = h5py.File(PATH+'X_test.hdf5', 'r')
    X_v = X_val.get('X_test')[:]
    y_val = h5py.File(PATH+'y_test.hdf5', 'r')
    y_v = y_val.get('y_test')[:]

    n = int(len(X_v)*0.5)
    X_va = X_v[:n]
    y_va = y_v[:n]
    X_t = X_v[n:]
    y_t = y_v[n:]

    # print(len(X_tr), len(X_va), len(X_t))

    return X_tr, y_tr, X_va, y_va, X_t, y_t
    # return X_v, y_v


def get_data(X, y, image_size=128):
    features, labels = [], []

    for i in range(len(X)):
      feature, label = load(X[i], y[i], i, image_size)
      features.append(feature.reshape(feature.shape[0], feature.shape[1], 1))
      labels.append(label.reshape(label.shape[0], label.shape[1], 1))

    return features, labels

def get_batches(X, y, batch_size):
    ''' Generación de batches '''

    count_inf = 0
    count_sup = batch_size

    n_batches = math.ceil(len(X)/batch_size)

    X_batches, y_batches = [], []

    for n in range(n_batches):

        X_batches.append(X[count_inf:count_sup])
        y_batches.append(y[count_inf:count_sup])

        count_inf = count_sup
        count_sup += batch_size

    return X_batches, y_batches

# read_data()
