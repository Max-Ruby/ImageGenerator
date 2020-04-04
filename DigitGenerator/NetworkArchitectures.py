from __future__ import print_function
from keras.models import Model
from keras.layers import Conv2D, Dense, Reshape, Conv2DTranspose, UpSampling2D, Input, Reshape, Concatenate, MaxPooling2D, ZeroPadding2D, Multiply
import tensorflow as tf; tf.image # There's a bug in tensorflow that requires us to import tf.image like this

'''
Various Notes:
Read ONEGENERATOR.md.
'''


def digit_generator(total_digit_dimension, num_digits):
    noise = Input(shape=(total_digit_dimension,))
    chosen_digit = Input(shape=(num_digits,))
    digit_classification = Dense(total_digit_dimension)(chosen_digit)
    restricted_noise = Multiply()([noise, digit_classification])
    coarse_1 = Reshape((1, 1, total_digit_dimension,))(restricted_noise)
    coarse_1 = ZeroPadding2D(padding=((3,3),(3,3)))(coarse_1)
    coarse_1 = Conv2D(total_digit_dimension, (5, 5), padding='same', activation='elu')(coarse_1)
    coarse_1 = Conv2D(32, (3, 3), padding='same', activation='elu')(coarse_1)
    coarse_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')(coarse_1)
    coarse_3 = UpSampling2D(size=(2, 2), interpolation='bilinear')(coarse_2)
    mid_2 = Conv2D(16, (3, 3), padding='same', activation='elu')(coarse_2)
    mid_2 = Conv2D(16, (3, 3), padding='same', activation='elu')(mid_2)
    mid_3 = UpSampling2D(size=(2, 2), interpolation='bilinear')(mid_2)
    fine_3 = Conv2D(8, (3, 3), padding='same', activation='elu')(mid_3)
    fine_3 = Conv2D(8, (3, 3), padding='same', activation='elu')(fine_3)
    composition = Concatenate(axis=3)([coarse_3, mid_3, fine_3])
    outputs = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(composition)
    model = Model(inputs=[noise,chosen_digit], outputs=outputs)
    return model


def digit_discriminator(num_digits):
    image = Input(shape=(28, 28, 1,))
    fine = Conv2D(32,(3,3), padding='same', activation='elu')(image)
    fine = Conv2D(32, (3, 3), padding='same', activation='elu')(fine)
    fine = Conv2D(32,(3,3), padding='same', activation='elu')(fine)
    mid = Conv2D(16,(3,3), padding='same', activation='elu', strides=(2,2))(fine)
    mid = Conv2D(16,(3,3), padding='same', activation='elu')(mid)
    mid = Conv2D(16,(3,3), padding='same', activation='elu')(mid)
    coarse = Conv2D(8,(3,3), padding='same', activation='elu', strides=(2,2))(mid)
    coarse = Conv2D(8,(3,3), padding='same', activation='elu')(coarse)
    coarse = Conv2D(8, (3, 3), padding='same', activation='elu')(coarse)
    fine_scores = MaxPooling2D(pool_size=(28,28))(fine)
    mid_scores = MaxPooling2D(pool_size=(14,14))(mid)
    coarse_scores= MaxPooling2D(pool_size=(7,7))(coarse)
    composition = Concatenate(axis=3)([fine_scores,mid_scores,coarse_scores])
    composition = Reshape((32+16+8,))(composition)
    outputs = Dense(num_digits, activation='sigmoid')(composition)
    model = Model(inputs=image, outputs=outputs)
    return model

def make_gan(generative_model, discriminator, total_digit_dimension, num_digits):
    discriminator.trainable = False  # Note: The discriminator being untrainable only applies to gan, not to the compiled discriminator above
    noise = Input(shape=(total_digit_dimension,))
    chosen_digit = Input(shape=(num_digits,))
    image = generative_model([noise,chosen_digit])
    prediction = discriminator(image)
    gan = Model(inputs=[noise,chosen_digit],outputs=prediction)
    return gan