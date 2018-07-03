'''
    ST-ResNet: Deep Spatio-temporal Residual Networks
'''

from __future__ import print_function
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Reshape,
    Conv2D,
    LSTM,
    Lambda, MaxPool2D, Flatten, Concatenate, GRU, Multiply)
from keras.layers.merge import Add
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.utils import plot_model

from deepst.models.mulLayer import mulLayer
from .iLayer import iLayer
from .sliceLayer import sliceLayer
import tensorflow as tf

output_nb_flow=2
stride=2
# from keras.utils.visualize_util import plot


def _shortcut(input, residual):
    # return merge([input, residual], mode='sum')
    return Add()([input, residual])


def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1), bn=False):
    def f(input):
        if bn:
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        # return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
        #                      border_mode="same")(activation)
        return Conv2D(strides=subsample, padding="same", kernel_size=(nb_row, nb_col), filters=nb_filter)(activation)

    return f


def _residual_unit(nb_filter, init_subsample=(1, 1)):
    def f(input):

        residual = _bn_relu_conv(nb_filter, stride, stride)(input)
        # residual = _bn_relu_conv(nb_filter, stride, stride)(residual)
        residual = _bn_relu_conv(nb_filter, stride, stride)(residual)
        return _shortcut(input, residual)

    return f


def ResUnits(residual_unit, nb_filter, repetations=1,pool=False):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            input = residual_unit(nb_filter=nb_filter,
                                  init_subsample=init_subsample)(input)
            if pool:
                input = MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(input)
        return input

    return f



def stresnet(c_conf=(3, 2, 32, 32), p_conf=(3, 2, 32, 32), t_conf=(3, 2, 32, 32), external_dim=8, nb_residual_unit=3):
    '''
    C - Temporal Closeness
    P - Period
    T - Trend
    conf = (len_seq, nb_flow, map_height, map_width)
    external_dim
    '''

    # main input
    main_inputs = []
    outputs = []

    # c_conf, p_conf, t_conf
    # for conf in [ c_conf]:
    #     if conf is not None:
    #         # base 模型
    #         len_seq, nb_flow, map_height, map_width = conf
    #         input = Input(shape=(nb_flow * len_seq, map_height, map_width))
    #         main_inputs.append(input)
    #         # Conv1
    #         # conv1 = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, border_mode="same")(input)
    #         conv1 = Conv2D(padding="same", kernel_size=(stride, stride), filters=64)(input)
    #         # [nb_residual_unit] Residual Units
    #         residual_output = ResUnits(_residual_unit, nb_filter=64,
    #                                    repetations=nb_residual_unit,)(conv1)
    #         # Conv2
    #         activation = Activation('relu')(residual_output)
    #         # conv2 = Convolution2D(
    #         #     nb_filter=nb_flow, nb_row=3, nb_col=3, border_mode="same")(activation)
    #         conv2 = Convolution2D(
    #             nb_filter=nb_flow, nb_row=3, nb_col=3, border_mode="same")(activation)
    #         outputs.append(conv2)

    for conf in [c_conf]:
        if conf is not None:
            # base 模型
            # len_seq, nb_flow, map_height, map_width = conf
            # input = Input(shape=(nb_flow * len_seq, map_height, map_width))
            # main_inputs.append(input)
            # # Conv1
            # # conv1 = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, border_mode="same")(input)
            # conv1 = Conv2D(padding="same", kernel_size=(3, 3), filters=64)(input)
            # # [nb_residual_unit] Residual Units
            # residual_output = ResUnits(_residual_unit, nb_filter=64,
            #                            repetations=nb_residual_unit)(conv1)
            # # Conv2
            # activation = Activation('relu')(residual_output)
            # # conv2 = Convolution2D(
            # #     nb_filter=nb_flow, nb_row=3, nb_col=3, border_mode="same")(activation)
            # conv2 = Convolution2D(
            #     nb_filter=output_nb_flow, nb_row=3, nb_col=3, border_mode="same")(activation)
            # outputs.append(conv2)
            # 我的模型
            len_seq, nb_flow, map_height, map_width = conf
            # input=main_inputs[0]
            input = Input(shape=(nb_flow * len_seq, map_height, map_width))
            # main_inputs[1:3]=main_inputs[0:2]
            # main_inputs[0]=input
            main_inputs.append(input)


            input = Reshape((len_seq, nb_flow, map_height, map_width))(input)
            timeSliceOutputs = []
            print(input.shape[1])
            # 定义共用的CNN模型
            # conv1=Reshape((nb_flow, map_height, map_width))(input[:,timeSlice])
            # Conv1
            aa = Conv2D(padding="same", kernel_size=(stride, stride), filters=64)
            # conv1 = Conv2D(padding="same", kernel_size=(3, 3), filters=64)
            # [nb_residual_unit] Residual Units
            nb_residual_unit = 5
            resIntput = Input(shape=(64, map_height, map_width))
            # resIntput=Input(shape=aa.output_shape)
            bb = ResUnits(_residual_unit, nb_filter=64,
                          repetations=nb_residual_unit, pool=True)(resIntput)
            resModel = Model(resIntput, bb)
            resModel.summary()
            plot_model(resModel, to_file='resModel.png')
            # Conv2
            cc = Activation('relu')
            # conv2 = Convolution2D(
            #     nb_filter=nb_flow, nb_row=3, nb_col=3, border_mode="same")(activation)
            dd = Convolution2D(
                nb_filter=128, nb_row=2, nb_col=2, border_mode="same")
            ee = Reshape((1, -1))
            for timeSlice in range(len_seq):
                ii = sliceLayer(1, timeSlice)(input)
                # ii=input[:, timeSlice]
                conv2 = ee(dd(cc(resModel(aa(ii)))))
                timeSliceOutputs.append(conv2)
            convOutput = Concatenate(axis=1)(timeSliceOutputs)
            # lstm = GRU(output_nb_flow * map_height * map_width)(convOutput)
            # lstm = GRU(2 * map_height * map_width)(convOutput)
            lstm = LSTM(600)(convOutput)
            # input+lstm
            out1 = sliceLayer(1, len_seq - 1)(input)
            def antirectifier(x):
                # x=np.reshape(x,[-1,18,2,64,64])
                # x = tf.reshape(x, [-1, 18, 2, 64, 64])
                # x = tf.reduce_sum(x, axis=1)
                x = K.reshape(x, [-1, 18, 2, map_height, map_width])
                x = K.sum(x, axis=1)
                return x
            def antirectifier_output_shape(input_shape):
                return (input_shape[0], 2, map_height, map_width)
            # out1 = Lambda(antirectifier,
            #               output_shape=antirectifier_output_shape)(out1)
            # out2 = Flatten()(lstm)
            sig = Dense(output_dim=nb_flow * map_height * map_width, activation='sigmoid')(lstm)
            sig = Reshape((nb_flow, map_height, map_width))(sig)
            tan = Dense(output_dim=nb_flow * map_height * map_width, activation='tanh')(lstm)
            tan = Reshape((nb_flow, map_height, map_width))(tan)
            den = Dense(output_dim=nb_flow * map_height * map_width)(lstm)
            den = Reshape((nb_flow, map_height, map_width))(den)
            # out2=Add()([Multiply()([outputs[0],sig]),tan,out1])
            mul=Multiply()([out1, sig])
            out2 = Add()([mul, den])

            # output = Add()([out1, tan])
            # outputs[0]=out2
            outputs.append(out2)


    # len_seq, nb_flow, map_height, map_width = c_conf
    # input = main_inputs[0]
    # # input = Input(shape=(nb_flow * len_seq, map_height, map_width))
    # input2 = Reshape((len_seq, nb_flow * map_height * map_width))(input)
    # lstm = LSTM(output_nb_flow * map_height * map_width)(input)
    # output = Reshape((output_nb_flow, map_height, map_width))(lstm)
    # outputs.append(output)
    # outputs=outputs[1:]

    # parameter-matrix-based fusion
    if len(outputs) == 1:
        main_output = outputs[0]
    else:

        new_outputs = []
        for output in outputs:
            # new_outputs.append(iLayer()(output))
            new_outputs.append(mulLayer()(output))
        main_output = Add()(new_outputs)
        # main_output=Add()(outputs)

    # fusing with external component
    # if external_dim != None and external_dim > 0:
    #     # external input
    #     external_input = Input(shape=(external_dim,))
    #     main_inputs.append(external_input)
    #     embedding = Dense(output_dim=10)(external_input)
    #     embedding = Activation('relu')(embedding)
    #     # h1 = Dense(output_dim=nb_flow * map_height * map_width)(embedding)
    #     h1 = Dense(output_dim=nb_flow * map_height * map_width)(embedding)
    #     activation = Activation('relu')(h1)
    #     # external_output = Reshape((nb_flow, map_height, map_width))(activation)
    #     external_output = Reshape((nb_flow, map_height, map_width))(activation)
    #     main_output = merge([main_output, external_output], mode='sum')
    # else:
    #     print('external_dim:', external_dim)

    main_output = Activation('tanh')(main_output)

    def antirectifier2(x):
        #x=np.reshape(x,[-1,18,2,64,64])
        x = K.reshape(x,[-1,18,2,map_height,map_width])
        x = K.sum(x, axis=1)
        return x

    # def antirectifier_output_shape(input_shape):
    #
    #     return (input_shape[0],2,64,64)
    #
    # main_output=Lambda(antirectifier2 )(main_output)
    model = Model(input=main_inputs, output=main_output)

    return model


if __name__ == '__main__':
    model = stresnet(external_dim=28, nb_residual_unit=12)
    # plot(model, to_file='ST-ResNet.png', show_shapes=True)
    model.summary()
