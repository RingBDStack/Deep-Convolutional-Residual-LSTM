from keras.models import Model
from keras.layers import *
from recurrentshop.engine import RNNCell

class LSTMCell2(RNNCell):
    def build_model(self, input_shape):
        output_dim = self.output_dim
        input_dim = input_shape[-1]
        output_shape = (input_shape[0], output_dim)
        x = Input(batch_shape=input_shape)

        c_tm1 = Input(batch_shape=output_shape)
        h_tm1 = Input(batch_shape=output_shape)
        c_tm2 = Input(batch_shape=output_shape)
        h_tm2 = Input(batch_shape=output_shape)

        c1 = add([
            multiply([c_tm1, Activation('sigmoid')(add([Dense(output_dim)(h_tm1), Dense(output_dim)(h_tm2)]))]),
            multiply([Dense(output_dim)(x),
                      Activation('sigmoid')(add([Dense(output_dim)(h_tm1), Dense(output_dim)(h_tm2)]))])
        ])
        h1 = multiply([h_tm1, Activation('sigmoid')(add([Dense(output_dim)(h_tm2), Dense(output_dim)(x)]))])
        c2 = add([
            multiply([c_tm2, Activation('sigmoid')(add([Dense(output_dim)(h_tm1), Dense(output_dim)(h_tm2)]))]),
            multiply([Dense(output_dim)(x),
                      Activation('sigmoid')(add([Dense(output_dim)(h_tm1), Dense(output_dim)(h_tm2)]))])
        ])
        h2 = multiply([h_tm2, Activation('sigmoid')(add([Dense(output_dim)(h_tm1), Dense(output_dim)(x)]))])
        out = Concatenate(axis=1)([c_tm1, c_tm2])
        return Model([x, c_tm1, h_tm1, c_tm2, h_tm2], [out, c1, h1, c2, h2])


class LSTMCell3(RNNCell):
    def build_model(self, input_shape):
        output_dim = self.output_dim
        input_dim = input_shape[-1]
        output_shape = (input_shape[0], output_dim)
        x = Input(batch_shape=input_shape)

        c_tm1 = Input(batch_shape=output_shape)
        h_tm1 = Input(batch_shape=output_shape)
        c_tm2 = Input(batch_shape=output_shape)
        h_tm2 = Input(batch_shape=output_shape)

        f1 = add(
            [Dense(output_dim)(x), Dense(output_dim, use_bias=False)(h_tm1), Dense(output_dim, use_bias=False)(h_tm2)])
        f1 = Activation('sigmoid')(f1)
        i1 = add(
            [Dense(output_dim)(x), Dense(output_dim, use_bias=False)(h_tm1), Dense(output_dim, use_bias=False)(h_tm2)])
        i1 = Activation('sigmoid')(i1)
        c_prime1 = add(
            [Dense(output_dim)(x), Dense(output_dim, use_bias=False)(h_tm1), Dense(output_dim, use_bias=False)(h_tm2)])
        c_prime1 = Activation('tanh')(c_prime1)
        c1 = add([multiply([f1, c_tm1]), multiply([i1, c_prime1])])
        c11 = Activation('tanh')(c1)
        o1 = add(
            [Dense(output_dim)(x), Dense(output_dim, use_bias=False)(h_tm1), Dense(output_dim, use_bias=False)(h_tm2)])
        o1 = Activation('sigmoid')(o1)
        h1 = multiply([o1, c11])

        f2 = add(
            [Dense(output_dim)(x), Dense(output_dim, use_bias=False)(h_tm1), Dense(output_dim, use_bias=False)(h_tm2)])
        f2 = Activation('sigmoid')(f2)
        i2 = add(
            [Dense(output_dim)(x), Dense(output_dim, use_bias=False)(h_tm1), Dense(output_dim, use_bias=False)(h_tm2)])
        i2 = Activation('sigmoid')(i2)
        c_prime2 = add(
            [Dense(output_dim)(x), Dense(output_dim, use_bias=False)(h_tm1), Dense(output_dim, use_bias=False)(h_tm2)])
        c_prime2 = Activation('tanh')(c_prime2)
        c2 = add([multiply([f2, c_tm2]), multiply([i2, c_prime2])])
        c22 = Activation('tanh')(c2)
        o2 = add(
            [Dense(output_dim)(x), Dense(output_dim, use_bias=False)(h_tm1), Dense(output_dim, use_bias=False)(h_tm2)])
        o2 = Activation('sigmoid')(o2)
        h2 = multiply([o2, c22])
        out = Concatenate(axis=1)([h1, h2])
        return Model([x, c_tm1, h_tm1, c_tm2, h_tm2], [out, c1, h1, c2, h2])

