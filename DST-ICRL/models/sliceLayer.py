from keras import backend as K
from keras.engine.topology import Layer
# from keras.layers import Dense
import tensorflow as tf
import numpy as np


class sliceLayer(Layer):
    def __init__(self, dim,index, **kwargs):
        self.dim=dim
        self.index = index
        super(sliceLayer, self).__init__(**kwargs)


    def call(self, x,**kwargs):
        if self.dim==1:
            x=x[:,self.index]
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[:self.dim]+input_shape[self.dim+1:]