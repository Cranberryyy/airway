#coding=utf-8
import tensorflow as tf
from tensorflow.keras.layers  import Conv3D, Conv3DTranspose, MaxPooling3D, concatenate, BatchNormalization, Activation, UpSampling3D, Input, Concatenate

def conv_block_nested(x, mid_ch, out_ch, kernel_size=3, padding='same', activation='relu'):
    x = Conv3D(mid_ch, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv3D(out_ch, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

class NestedUNet():
    def __init__(self, init_channel=8, num_of_output=2, activation='relu', deep_supervision=False):
        self.init_channel = init_channel
        self.num_of_output = num_of_output
        self.deep_supervision = deep_supervision
        self.activation = activation
    
    def __call__(self, x, name='nestedunet3d'):

        t = 2
        n1 = self.init_channel
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        x0_0 = conv_block_nested(x, filters[0], filters[0], activation=self.activation)

        x1_0 = conv_block_nested(MaxPooling3D(strides=2)(x0_0), filters[1], filters[1], activation=self.activation)
        x0_1 = conv_block_nested(Concatenate()([x0_0, UpSampling3D()(x1_0)]), filters[0], filters[0], activation=self.activation)

        x2_0 = conv_block_nested(MaxPooling3D(strides=2)(x1_0), filters[2], filters[2], activation=self.activation)
        x1_1 = conv_block_nested(Concatenate()([x1_0, UpSampling3D()(x2_0)]), filters[1], filters[1], activation=self.activation)
        x0_2 = conv_block_nested(Concatenate()([x0_0, x0_1, UpSampling3D()(x1_1)]), filters[0], filters[0], activation=self.activation)

        x3_0 = conv_block_nested(MaxPooling3D(strides=2)(x2_0), filters[3], filters[3], activation=self.activation)
        x2_1 = conv_block_nested(Concatenate()([x2_0, UpSampling3D()(x3_0)]), filters[2], filters[2], activation=self.activation)
        x1_2 = conv_block_nested(Concatenate()([x1_0, x1_1, UpSampling3D()(x2_1)]), filters[1], filters[1], activation=self.activation)
        x0_3 = conv_block_nested(Concatenate()([x0_0, x0_1, x0_2, UpSampling3D()(x1_2)]), filters[0], filters[0], activation=self.activation)

        x4_0 = conv_block_nested(MaxPooling3D(strides=2)(x3_0), filters[4], filters[4], activation=self.activation)
        x3_1 = conv_block_nested(Concatenate()([x3_0, UpSampling3D()(x4_0)]), filters[3], filters[3], activation=self.activation)
        x2_2 = conv_block_nested(Concatenate()([x2_0, x2_1, UpSampling3D()(x3_1)]), filters[2], filters[2], activation=self.activation)
        x1_3 = conv_block_nested(Concatenate()([x1_0, x1_1, x1_2, UpSampling3D()(x2_2)]), filters[1], filters[1], activation=self.activation)
        x0_4 = conv_block_nested(Concatenate()([x0_0, x0_1, x0_2, x0_3, UpSampling3D()(x1_3)]), filters[0], filters[0], activation=self.activation)

        if not isinstance(self.num_of_output, list):
            output = Conv3D(self.num_of_output, 3, activation = 'softmax', name=name+'conv10_1')(x0_4)
        else:
            output_2 = Conv3D(self.num_of_output[0], 1, activation = 'softmax', name=name+'conv10_1_0')(x0_4)
            soft_argmax = tf.nn.softmax(output_2, axis=-1)
            output_27 = Conv3D(self.num_of_output[1], 3, activation = 'softmax', name=name+'conv10_1_1')(soft_argmax)          
            output = [output_2, output_27]
        return output
