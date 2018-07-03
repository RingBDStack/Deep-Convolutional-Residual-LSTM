# import numpy as np
from keras import backend as K


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))


def root_mean_square_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def rmseIn(y_true, y_pred):
    shape = y_pred.shape
    shape2=[-1 , int(int(shape[1]) / 2) , 2]
    shape2.extend(shape[2:])
    shape2=[int(i) for i in shape2]
    y_true = K.reshape(y_true, shape2)
    y_pred = K.reshape(y_pred, shape2)
    return mean_squared_error(y_true[:, :,0], y_pred[:,:, 0]) ** 0.5


def rmseOut(y_true, y_pred):
    shape = y_pred.shape
    shape2 = [-1, int(int(shape[1]) / 2), 2]
    shape2.extend(shape[2:])
    shape2 = [int(i) for i in shape2]
    y_true = K.reshape(y_true, shape2)
    y_pred = K.reshape(y_pred, shape2)
    return mean_squared_error(y_true[:,:, 1], y_pred[:,:, 1]) ** 0.5


# aliases
mse = MSE = mean_squared_error


# rmse = RMSE = root_mean_square_error


def masked_mean_squared_error(y_true, y_pred):
    idx = (y_true > 1e-6).nonzero()
    return K.mean(K.square(y_pred[idx] - y_true[idx]))


def masked_rmse(y_true, y_pred):
    return masked_mean_squared_error(y_true, y_pred) ** 0.5
