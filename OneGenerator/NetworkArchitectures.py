from __future__ import print_function
from keras.models import Model
from keras.layers import Conv2D, Dense, Reshape, Conv2DTranspose, UpSampling2D, Input, Reshape, Concatenate, MaxPooling2D, ZeroPadding2D
import tensorflow as tf; tf.image # There's a bug in tensorflow that requires us to import tf.image like this


def one_generator(digit_dimension):
    inputs = Input(shape=(digit_dimension,))
    coarse_1 = Reshape((1, 1, digit_dimension,))(inputs)
    coarse_1 = ZeroPadding2D(padding=((3,3),(3,3)))(coarse_1)
    coarse_1 = Conv2D(8, (7, 7), padding='same', activation='elu')(coarse_1)
    coarse_1 = Conv2D(8, (3, 3), padding='same', activation='elu')(coarse_1)
    coarse_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')(coarse_1)
    coarse_3 = UpSampling2D(size=(2, 2), interpolation='bilinear')(coarse_2)
    mid_2 = Conv2D(8, (3, 3), padding='same', activation='elu')(coarse_2)
    mid_2 = Conv2D(8, (3, 3), padding='same', activation='elu')(mid_2)
    mid_3 = UpSampling2D(size=(2, 2), interpolation='bilinear')(mid_2)
    fine_3 = Conv2D(8, (3, 3), padding='same', activation='elu')(mid_3)
    fine_3 = Conv2D(8, (3, 3), padding='same', activation='elu')(fine_3)
    composition = Concatenate(axis=3)([coarse_3, mid_3, fine_3])
    outputs = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(composition)
    model = Model(inputs=inputs, outputs=outputs)

    return model


def one_discriminator():
    inputs = Input(shape=(28, 28, 1,))
    fine = Conv2D(4,(3,3), padding='same', activation='elu')(inputs)
    fine = Conv2D(4,(3,3), padding='same', activation='elu')(fine)
    mid = Conv2D(4,(3,3), padding='same', activation='elu', strides=(2,2))(fine)
    mid = Conv2D(4,(3,3), padding='same', activation='elu')(mid)
    coarse = Conv2D(4,(3,3), padding='same', activation='elu', strides=(2,2))(mid)
    coarse = Conv2D(4,(3,3), padding='same', activation='elu')(coarse)
    fine_scores = MaxPooling2D(pool_size=(28,28))(fine)
    mid_scores = MaxPooling2D(pool_size=(14,14))(mid)
    coarse_scores= MaxPooling2D(pool_size=(7,7))(coarse)
    composition = Concatenate(axis=3)([fine_scores,mid_scores,coarse_scores])
    composition = Reshape((12,))(composition)
    outputs = Dense(1, activation='sigmoid')(composition)
    model = Model(inputs=inputs, outputs=outputs)
    return model
