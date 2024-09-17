# coding=utf-8
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling3D, Conv3D, MaxPooling3D, Dense,  BatchNormalization, \
                    Reshape, multiply, Activation, concatenate, UpSampling3D



def SEModule(input, ratio, out_dim, activation='relu'):
    # bs, c, h, w
    x = GlobalAveragePooling3D()(input)
    excitation = Dense(units=out_dim // ratio)(x)
    excitation = Activation(activation)(excitation)
    excitation = Dense(units=out_dim)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1, 1, 1, out_dim))(excitation)
    scale = multiply([input, excitation])
    return scale

class SEUnet3D():
    def __init__(self, init_channel=8, num_of_output=2, activation='relu', deep_supervision=False):
        self.init_channel = init_channel
        self.num_of_output = num_of_output
        self.deep_supervision = deep_supervision
        self.activation = activation
    
    def __call__(self, inputs, name='seunet3d'):
        conv1 = Conv3D(16,
                    3,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(inputs)
        conv1 = BatchNormalization()(conv1)

        conv1 = Conv3D(16,
                    3,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(conv1)
        conv1 = BatchNormalization()(conv1)

        # se
        conv1 = SEModule(conv1, 4, 16)

        pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
        conv2 = Conv3D(32,
                    3,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(pool1)
        conv2 = BatchNormalization()(conv2)

        conv2 = Conv3D(32,
                    3,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(conv2)
        conv2 = BatchNormalization()(conv2)

        # se
        conv2 = SEModule(conv2, 8, 32)

        pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
        conv3 = Conv3D(64,
                    3,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(pool2)
        conv3 = BatchNormalization()(conv3)

        conv3 = Conv3D(64,
                    3,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(conv3)
        conv3 = BatchNormalization()(conv3)

        # se
        conv3 = SEModule(conv3, 8, 64)

        pool3 = MaxPooling3D(pool_size=(2, 2,2))(conv3)
        conv4 = Conv3D(128,
                    3,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(pool3)
        conv4 = BatchNormalization()(conv4)

        conv4 = Conv3D(128,
                    3,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(conv4)
        conv4 = BatchNormalization()(conv4)

        # se
        conv4 = SEModule(conv4, 16, 128)

        pool4 = MaxPooling3D(pool_size=(2, 2,2))(conv4)

        conv5 = Conv3D(256,
                    3,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv3D(256,
                    3,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(conv5)
        conv5 = BatchNormalization()(conv5)

        # se
        conv5 = SEModule(conv5, 16, 256)

        up6 = Conv3D(128,
                    2,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(UpSampling3D(size=(2,2, 2))(conv5))
        up6 = BatchNormalization()(up6)

        merge6 = concatenate([conv4, up6], axis=3)
        conv6 = Conv3D(128,
                    3,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(merge6)
        conv6 = BatchNormalization()(conv6)

        conv6 = Conv3D(128,
                    3,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(conv6)
        conv6 = BatchNormalization()(conv6)

        # se
        conv6 = SEModule(conv6, 16, 128)

        up7 = Conv3D(64,
                    2,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(UpSampling3D(size=(2,2,2))(conv6))
        up7 = BatchNormalization()(up7)

        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv3D(64,
                    3,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(merge7)
        conv7 = BatchNormalization()(conv7)

        conv7 = Conv3D(64,
                    3,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(conv7)
        conv7 = BatchNormalization()(conv7)

        # se
        conv7 = SEModule(conv7, 8, 64)

        up8 = Conv3D(32,
                    2,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(UpSampling3D(size=(2,2,2))(conv7))
        up8 = BatchNormalization()(up8)

        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv3D(32,
                    3,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(merge8)
        conv8 = BatchNormalization()(conv8)

        conv8 = Conv3D(32,
                    3,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(conv8)
        conv8 = BatchNormalization()(conv8)

        # se
        conv8 = SEModule(conv8, 4, 32)

        up9 = Conv3D(16,
                    2,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(UpSampling3D(size=(2,2,2))(conv8))
        up9 = BatchNormalization()(up9)

        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv3D(16,
                    3,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(merge9)
        conv9 = BatchNormalization()(conv9)

        conv9 = Conv3D(16,
                    3,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(conv9)
        conv9 = BatchNormalization()(conv9)

        # se
        conv9 = SEModule(conv9, 2, 16)
        if not isinstance(self.num_of_output, list):
            output = Conv3D(self.num_of_output, 1, activation = 'softmax', name=name+'conv10_1')(conv9)
        else:
            output_2 = Conv3D(self.num_of_output[0], 1, activation = 'softmax', name=name+'conv10_1_0')(conv9)
            soft_argmax = tf.nn.softmax(output_2, axis=-1)
            output_27 = Conv3D(self.num_of_output[1], 3, activation = 'softmax', name=name+'conv10_1_1')(soft_argmax)          
            output = [output_2, output_27]
        return output