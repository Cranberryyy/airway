import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv3D, BatchNormalization , MaxPooling3D, Dropout, concatenate, UpSampling3D, Activation
import tensorflow_addons as tfa
from .ops import distance_transform, boundary_locate

def Conv3DLayer(x, kernel_num, kernel_size, activation=None, padding='same', name='conv', is_bn=False):
    if is_bn:
        x = Conv3D(kernel_num, kernel_size, activation=activation, padding=padding, name=name)(x)
        return tfa.layers.InstanceNormalization(name=name+'_instnorm')(x)
        # return BatchNormalization(name=name+'_instnorm')(x)
    else:
        return Conv3D(kernel_num, kernel_size, activation=activation, padding=padding, name=name)(x)


class Unet3D():
    def __init__(self, init_channel=8, num_of_output=2, activation='relu', deep_supervision=False, 
                 is_bn=False):
        self.kernel_num = np.array([1, 2, 4, 8, 16]) * init_channel
        self.num_of_output = num_of_output
        self.deep_supervision = deep_supervision
        self.activation = activation
        self.is_bn = is_bn

    def __call__(self, x, is_mid_feat_in=False, is_mid_feat_out=False, is_distance=False, name='unet3d'):
        if not is_mid_feat_in:
            conv1 = Conv3DLayer(x, self.kernel_num[0], 3, activation = self.activation, padding = 'same', name=name+'conv1_1', is_bn=self.is_bn)
            conv1 = Conv3DLayer(conv1, self.kernel_num[0], 3, activation = self.activation, padding = 'same', name=name+'conv1_2', is_bn=self.is_bn)
            pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
            conv2 = Conv3DLayer(pool1, self.kernel_num[1], 3, activation = self.activation, padding = 'same', name=name+'conv2_1', is_bn=self.is_bn)
            conv2 = Conv3DLayer(conv2, self.kernel_num[1], 3, activation = self.activation, padding = 'same', name=name+'conv2_2', is_bn=self.is_bn)
            pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
            conv3 = Conv3DLayer(pool2, self.kernel_num[2], 3, activation = self.activation, padding = 'same', name=name+'conv3_1', is_bn=self.is_bn)
            conv3 = Conv3DLayer(conv3, self.kernel_num[2], 3, activation = self.activation, padding = 'same', name=name+'conv3_2', is_bn=self.is_bn)
            pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
            conv4 = Conv3DLayer(pool3, self.kernel_num[3], 3, activation = self.activation, padding = 'same', name=name+'conv4_1', is_bn=self.is_bn)
            conv4 = Conv3DLayer(conv4, self.kernel_num[3], 3, activation = self.activation, padding = 'same', name=name+'conv4_2', is_bn=self.is_bn)
            drop4 = conv4 #Dropout(0.1)(conv4)
            pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)

            conv5 = Conv3DLayer(pool4, self.kernel_num[4], 3, activation = self.activation, padding = 'same', name=name+'conv5_1', is_bn=self.is_bn)
            conv5 = Conv3DLayer(conv5, self.kernel_num[4], 3, activation = self.activation, padding = 'same', name=name+'conv5_2', is_bn=self.is_bn)
            drop5 = conv5 # Dropout(0.1)(conv5)
        else:
            conv1, conv2, conv3, drop4, drop5 = x

        up6 = Conv3DLayer(UpSampling3D(size = (2,2,2))(drop5), self.kernel_num[3], 3, activation = self.activation, padding = 'same', name=name+'conv6_1', is_bn=self.is_bn)
        merge6 = concatenate([drop4,up6],axis=-1)
        conv6 = Conv3DLayer(merge6, self.kernel_num[3], 3, activation = self.activation, padding = 'same', name=name+'conv6_2', is_bn=self.is_bn)
        conv6 = Conv3DLayer(conv6, self.kernel_num[3], 3, activation = self.activation, padding = 'same', name=name+'conv6_3', is_bn=self.is_bn)

        up7 = Conv3DLayer(UpSampling3D(size = (2,2,2))(conv6), self.kernel_num[2], 3, activation = self.activation, padding = 'same', name=name+'conv7_1', is_bn=self.is_bn)
        merge7 = concatenate([conv3,up7],axis=-1)
        conv7 = Conv3DLayer(merge7, self.kernel_num[2], 3, activation = self.activation, padding = 'same', name=name+'conv7_2', is_bn=self.is_bn)
        conv7 = Conv3DLayer(conv7, self.kernel_num[2], 3, activation = self.activation, padding = 'same', name=name+'conv7_3', is_bn=self.is_bn)

        up8 = Conv3DLayer(UpSampling3D(size = (2,2,2))(conv7), self.kernel_num[1], 3, activation = self.activation, padding = 'same', name=name+'conv8_1', is_bn=self.is_bn)
        merge8 = concatenate([conv2,up8],axis=-1)
        conv8 = Conv3DLayer(merge8, self.kernel_num[1], 3, activation = self.activation, padding = 'same', name=name+'conv8_2', is_bn=self.is_bn)
        conv8 = Conv3DLayer(conv8, self.kernel_num[1], 3, activation = self.activation, padding = 'same', name=name+'conv8_3', is_bn=self.is_bn)

        up9 = Conv3DLayer(UpSampling3D(size = (2,2,2))(conv8), self.kernel_num[0], 3, activation = self.activation, padding = 'same', name=name+'conv9_1', is_bn=self.is_bn)
        merge9 = concatenate([conv1,up9],axis=-1)
        conv9 = Conv3DLayer(merge9, self.kernel_num[0], 3, activation = self.activation, padding = 'same', name=name+'conv9_2', is_bn=self.is_bn)
        conv9 = Conv3DLayer(conv9, self.kernel_num[0], 3, activation = self.activation, padding = 'same', name=name+'conv9_3', is_bn=self.is_bn)

        if not isinstance(self.num_of_output, list):
            if self.num_of_output == 1:
                output = [Conv3D(self.num_of_output, 1, activation = 'sigmoid', padding = 'same', name=name+'conv10_1')(conv9)]
            else:
                output = Conv3D(self.num_of_output, 1, activation = 'softmax', padding = 'same', name=name+'conv10_1')(conv9)
                if is_distance:
                    distance = boundary_locate(output)
                    output = [output] + distance
        else:
            output_2 = Conv3D(self.num_of_output[0], 1, activation = 'softmax', padding = 'same', name=name+'conv10_1_0')(conv9)
            output_27 = Conv3D(self.num_of_output[1], 3, activation = 'sigmoid', padding = 'same', name=name+'conv_connectivity')(output_2) 
            output = [output_2, output_27] 
            if is_distance:
                distance = boundary_locate(output_2)
                output = output + distance

        if self.deep_supervision:
            output_conv7 = Conv3D(1, 1, activation = 'sigmoid', padding = 'same', name=name+'conv7_deep')(UpSampling3D(size=(4,4,4))(conv7))
            output_conv8 = Conv3D(1, 1, activation = 'sigmoid', padding = 'same', name=name+'conv8_deep')(UpSampling3D(size=(2,2,2))(conv8))
            output = output+[output_conv8, output_conv7]
        
        if is_mid_feat_out:
            mid_feature = [conv1, conv2, conv3, conv4, conv5]
            return output, mid_feature
        else:
            return output