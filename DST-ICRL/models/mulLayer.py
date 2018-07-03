from keras import backend as K
from keras.engine.topology import Layer
# from keras.layers import Dense
import tensorflow as tf
import numpy as np


class mulLayer(Layer):
    def __init__(self, **kwargs):

        super(mulLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='aaa',
            shape=(1,1),
            initializer='uniform',
            trainable=True)
        self.built = True

    def call(self, x):
        return x * self.W

    def compute_output_shape(self, input_shape):
        return input_shape
