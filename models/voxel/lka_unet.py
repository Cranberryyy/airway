#coding=utf-8
import tensorflow as tf
from tensorflow.keras.layers  import Conv3D, MaxPooling3D,  Add, \
            BatchNormalization, Activation, UpSampling3D, concatenate
import tensorflow_addons as tfa
from .ops import distance_transform, boundary_locate
import numpy as np

def Conv3DLayer(x, kernel_num, kernel_size, activation=None, padding='same', name='conv', is_bn=False):
    if is_bn:
        x = Conv3D(kernel_num, kernel_size, activation=activation, padding=padding, name=name)(x)
        return tfa.layers.InstanceNormalization(name=name+'_instnorm')(x)
        # return BatchNormalization(name=name+'_instnorm')(x)
    else:
        return Conv3D(kernel_num, kernel_size, activation=activation, padding=padding, name=name)(x)


def large_kernel_attention(inputs, block_n_filter,  activation='relu', split_num=4, name='lka'):
    k_sizes = [[1,3,5], [1,5,7],[1,3,5],[1,5,7]] #[[3, 5, 1], [5, 7,1], [7, 9, 1]]
    dilation_rates = [1,1,1,1]
    if isinstance(inputs, list):
        for i in range(len(inputs)):
            x = inputs[i]
            if x.shape[-1] != block_n_filter:
                x = Conv3D(block_n_filter, 1, padding='same', name=name+'_norm_c_%d' %(i))(x)
            inputs[i] = x
        input = Add()(inputs)
    else:
        input = Conv3D(block_n_filter, 1, padding='same', name=name+'_norm_c')(inputs)

    atten_input = Conv3D(block_n_filter, 1)(input)
    input_list = tf.split(atten_input, split_num, axis=-1)
    scaled_output = []; res_output = []
    n_filter = block_n_filter

    def large_kernel_attention(input, k_size, n_feats, dilation_rate, name):
        output = Conv3DLayer(input, n_feats, k_size[0], padding='same', activation=activation, name=name+'_conv1')
        output = Conv3DLayer(output, n_feats, k_size[1], padding='same',  activation=activation, name=name+'_conv2')
        output = Conv3DLayer(output, n_feats, k_size[2], padding='same',  activation=activation, name=name+'_conv3')
        return output

    for i in range(split_num):
        temp = large_kernel_attention(input_list[i], k_sizes[i], n_filter,  dilation_rates[i], name="%s_upper_split_%d" %(name, i))
        scaled_output.append(temp)
    for i in range(split_num):
        temp = Conv3D(n_filter, k_sizes[0][0], padding='same', name="%s_lower_split_%d" %(name, i))(input_list[i])
        res_output.append(temp)
    atten = tf.concat([res_output[i] * scaled_output[i] for i in range(split_num)], axis=-1)
    return atten




class LKA_UNet():
    def __init__(self, init_channel=8, num_of_output=2, activation='relu', deep_supervision=False, 
                 is_bn=False):
        self.kernel_num = np.array([1, 2, 4, 8, 16]) * init_channel
        self.num_of_output = num_of_output
        self.deep_supervision = deep_supervision
        self.activation = activation
        self.is_bn = is_bn
        self.block_n_filters = self.kernel_num   #[6, 12, 18, 24]

    def __call__(self, x, is_mid_feat_in=False, is_mid_feat_out=False, is_distance=False, is_merge=False, name='unet3d'):
        if  is_merge:
            merge = Conv3DLayer(x, 8, 3, activation = self.activation, padding = 'same', name='merge_conv1_1')
            merge = Conv3DLayer(merge, 2, 3, activation = self.activation, padding = 'same', name='merge_conv1_2')   
            return merge

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
            pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

            conv5 = Conv3DLayer(pool4, self.kernel_num[4], 3, activation = self.activation, padding = 'same', name=name+'conv5_1', is_bn=self.is_bn)
            conv5 = Conv3DLayer(conv5, self.kernel_num[4], 3, activation = self.activation, padding = 'same', name=name+'conv5_2', is_bn=self.is_bn)
        else:
            conv1, conv2, conv3, conv4, conv5 = x
        atten_feat = []
        up6 = Conv3DLayer(UpSampling3D(size = (2,2,2))(conv5), self.kernel_num[3], 3, activation = self.activation, padding = 'same', name=name+'conv6_1', is_bn=self.is_bn)
        conv4_1 = large_kernel_attention(conv4, self.block_n_filters[3],  activation=self.activation, split_num=4, name=name+'lka4')
        atten_feat.append(conv4_1)
        merge6 = concatenate([conv4_1,up6],axis=-1)
        conv6 = Conv3DLayer(merge6, self.kernel_num[3], 3, activation = self.activation, padding = 'same', name=name+'conv6_2', is_bn=self.is_bn)
        conv6 = Conv3DLayer(conv6, self.kernel_num[3], 3, activation = self.activation, padding = 'same', name=name+'conv6_3', is_bn=self.is_bn)

        up7 = Conv3DLayer(UpSampling3D(size = (2,2,2))(conv6), self.kernel_num[2], 3, activation = self.activation, padding = 'same', name=name+'conv7_1', is_bn=self.is_bn)
        conv3_1 = large_kernel_attention(conv3, self.block_n_filters[2],  activation=self.activation, split_num=4, name=name+'lka3')
        atten_feat.append(conv3_1)
        merge7 = concatenate([conv3_1,up7],axis=-1)
        conv7 = Conv3DLayer(merge7, self.kernel_num[2], 3, activation = self.activation, padding = 'same', name=name+'conv7_2', is_bn=self.is_bn)
        conv7 = Conv3DLayer(conv7, self.kernel_num[2], 3, activation = self.activation, padding = 'same', name=name+'conv7_3', is_bn=self.is_bn)

        up8 = Conv3DLayer(UpSampling3D(size = (2,2,2))(conv7), self.kernel_num[1], 3, activation = self.activation, padding = 'same', name=name+'conv8_1', is_bn=self.is_bn)
        conv2_1 = large_kernel_attention(conv2, self.block_n_filters[1],  activation=self.activation, split_num=4, name=name+'lka2')
        atten_feat.append(conv2_1)
        merge8 = concatenate([conv2_1,up8],axis=-1)
        conv8 = Conv3DLayer(merge8, self.kernel_num[1], 3, activation = self.activation, padding = 'same', name=name+'conv8_2', is_bn=self.is_bn)
        conv8 = Conv3DLayer(conv8, self.kernel_num[1], 3, activation = self.activation, padding = 'same', name=name+'conv8_3', is_bn=self.is_bn)

        up9 = Conv3DLayer(UpSampling3D(size = (2,2,2))(conv8), self.kernel_num[0], 3, activation = self.activation, padding = 'same', name=name+'conv9_1', is_bn=self.is_bn)
        conv1_1 = large_kernel_attention(conv1, self.block_n_filters[0],  activation=self.activation, split_num=4, name=name+'lka1')
        atten_feat.append(conv1_1)
        merge9 = concatenate([conv1_1,up9],axis=-1)
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
        


         
