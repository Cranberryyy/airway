import tensorflow as tf
from tensorflow.keras.layers import Conv3D, UpSampling3D, concatenate, BatchNormalization, Activation, add
from tensorflow.keras.optimizers import *
import tensorflow_addons as tfa
from .ops import distance_transform

def batch_Norm_Activation(x, BN=False, name='bn'): ## To Turn off Batch Normalization, Change BN to False >
    if BN == True:
        x= tfa.layers.InstanceNormalization(name=name)(x)
        #x = BatchNormalization(name=name)(x)
        x = Activation("relu")(x)
    else:
        x= Activation("relu")(x)
    return x

class ResUnet():
    def __init__(self, init_channel=8, num_of_output=2, activation='relu', deep_supervision=False, 
                 is_bn=False):
        self.init_channel = init_channel
        self.num_of_output = num_of_output
        self.deep_supervision = deep_supervision
        self.activation = activation
        self.is_bn = is_bn

    def __call__(self, inputs, is_mid_feat_in=False, is_mid_feat_out=False, is_distance=False, name='resunet_3d'):
        filters = self.init_channel
        # Encoder
        if not is_mid_feat_in:       
            conv = Conv3D(filters*1, 3, activation = self.activation, padding = 'same', strides=1, name=name+'conv1_0')(inputs)
            conv = batch_Norm_Activation(conv, BN=self.is_bn, name=name+'bn1_0')
            conv = Conv3D(filters*1, 3, activation = self.activation, padding = 'same', strides=1, name=name+'conv1_1')(conv)
            shortcut = Conv3D(filters*1, 1, padding='same', strides=1, name=name+'conv1_2')(inputs)
            shortcut = batch_Norm_Activation(shortcut, BN=self.is_bn, name=name+'bn1_1')
            output1 = add([conv, shortcut])
            
            res1 = batch_Norm_Activation(output1, BN=self.is_bn, name=name+'bn2_0')
            res1 = Conv3D(filters*2, 3, activation = self.activation, padding = 'same', strides=2, name=name+'conv2_0')(res1)
            res1 = batch_Norm_Activation(res1, BN=self.is_bn, name=name+'bn2_1')
            res1 = Conv3D(filters*2, 3, activation = self.activation, padding = 'same', strides=1, name=name+'conv2_1')(res1)
            shortcut1 = Conv3D(filters*2, 3, padding='same', strides=(2,2,2), name=name+'conv2_2')(output1)
            shortcut1 = batch_Norm_Activation(shortcut1, BN=self.is_bn, name=name+'bn2_2')
            output2 = add([shortcut1, res1])
            
            res2 = batch_Norm_Activation(output2, BN=self.is_bn, name=name+'bn3_0')
            res2 = Conv3D(filters*4, 3, activation = self.activation, padding = 'same', strides=2, name=name+'conv3_0')(res2)
            res2 = batch_Norm_Activation(res2, BN=self.is_bn, name=name+'bn3_1')
            res2 = Conv3D(filters*4, 3, activation = self.activation, padding = 'same', strides=1, name=name+'conv3_1')(res2)
            shortcut2 = Conv3D(filters*4, 3, padding='same', strides=(2,2,2), name=name+'conv3_2')(output2)
            shortcut2 = batch_Norm_Activation(shortcut2, BN=self.is_bn, name=name+'bn3_2')
            output3 = add([shortcut2, res2])
        
            res3 = batch_Norm_Activation(output3, BN=self.is_bn, name=name+'bn4_0')
            res3 = Conv3D(filters*8, 3, activation = self.activation, padding = 'same', strides=2, name=name+'conv4_0')(res3)
            res3 = batch_Norm_Activation(res3, BN=self.is_bn, name=name+'bn4_1')
            res3 = Conv3D(filters*8, 3, activation = self.activation, padding = 'same', strides=1, name=name+'conv4_1')(res3)
            shortcut3 = Conv3D(filters*8, 3, padding='same', strides=(2,2,2), name=name+'conv4_2')(output3)
            shortcut3 = batch_Norm_Activation(shortcut3, BN=self.is_bn, name=name+'bn4_2')
            output4 = add([shortcut3, res3])
        
            res4 = batch_Norm_Activation(output4, BN=self.is_bn, name=name+'bn5_0')
            res4 = Conv3D(filters*16, 3, activation = self.activation, padding = 'same', strides=2, name=name+'conv5_0')(res4)
            res4 = batch_Norm_Activation(res4, BN=self.is_bn, name=name+'bn5_1')
            res4 = Conv3D(filters*16, 3, activation = self.activation, padding = 'same', strides=1, name=name+'conv5_1')(res4)
            shortcut4 = Conv3D(filters*16, 3, padding='same', strides=(2,2,2), name=name+'conv5_2')(output4)
            shortcut4 = batch_Norm_Activation(shortcut4, BN=self.is_bn, name=name+'bn5_2')
            output5 = add([shortcut4, res4])
        else:
            output1, output2, output3, output4, output5 = inputs
        
        #bridge
        conv = batch_Norm_Activation(output5, BN=self.is_bn, name=name+'bn6_0')
        conv = Conv3D(filters*16, 3, activation = self.activation, padding = 'same', strides=1, name=name+'conv6_0')(conv)
        conv = batch_Norm_Activation(conv, BN=self.is_bn, name=name+'bn6_1')
        conv = Conv3D(filters*16, 3, activation = self.activation, padding = 'same', strides=1, name=name+'conv6_1')(conv)
        
        #decoder
    
        uconv1 = UpSampling3D((2,2,2))(conv)
        uconv1 = concatenate([uconv1, output4])
        
        uconv11 = batch_Norm_Activation(uconv1, BN=self.is_bn, name=name+'bn11_0')
        uconv11 = Conv3D(filters*16, 3, activation = self.activation, padding = 'same', strides=1, name=name+'con11_0')(uconv11)
        uconv11 = batch_Norm_Activation(uconv11, BN=self.is_bn, name=name+'bn11_1')
        uconv11 = Conv3D(filters*16, 3, activation = self.activation, padding = 'same', strides=1, name=name+'conv11_1')(uconv11)
        shortcut5 = Conv3D(filters*16, 3, padding='same', strides=1, name=name+'conv11_2')(uconv1)
        shortcut5 = batch_Norm_Activation(shortcut5, BN=self.is_bn, name=name+'bn11_2')
        output6 = add([uconv11,shortcut5])
    
        uconv2 = UpSampling3D((2,2,2))(output6)
        uconv2 = concatenate([uconv2, output3])
        
        uconv22 = batch_Norm_Activation(uconv2, BN=self.is_bn, name=name+'bn22_0')
        uconv22 = Conv3D(filters*8, 3, activation = self.activation, padding = 'same', strides=1, name=name+'conv22_0')(uconv22)
        uconv22 = batch_Norm_Activation(uconv22, BN=self.is_bn, name=name+'bn22_1')
        uconv22 = Conv3D(filters*8, 3, activation = self.activation, padding = 'same', strides=1, name=name+'conv22_1')(uconv22)
        shortcut6 = Conv3D(filters*8, 3, padding='same', strides=1, name=name+'conv22_2')(uconv2)
        shortcut6 = batch_Norm_Activation(shortcut6, BN=self.is_bn, name=name+'bn22_2')
        output7 = add([uconv22,shortcut6])
        
        uconv3 = UpSampling3D((2,2,2))(output7)
        uconv3 = concatenate([uconv3, output2])
    
        uconv33 = batch_Norm_Activation(uconv3, BN=self.is_bn, name=name+'bn33_0')
        uconv33 = Conv3D(filters*4, 3, activation = self.activation, padding = 'same', strides=1, name=name+'conv33_0')(uconv33)
        uconv33 = batch_Norm_Activation(uconv33, BN=self.is_bn, name=name+'bn33_1')
        uconv33 = Conv3D(filters*4, 3, activation = self.activation, padding = 'same', strides=1, name=name+'conv33_1')(uconv33)
        shortcut7 = Conv3D(filters*4, 3, padding='same', strides=1, name=name+'conv33_2')(uconv3)
        shortcut7 = batch_Norm_Activation(shortcut7, BN=self.is_bn, name=name+'bn33_2')
        output8 = add([uconv33,shortcut7])
        
        uconv4 = UpSampling3D((2,2,2))(output8)
        uconv4 = concatenate([uconv4, output1])
        
        uconv44 = batch_Norm_Activation(uconv4, BN=self.is_bn, name=name+'bn44_0')
        uconv44 = Conv3D(filters*2, 3, activation = self.activation, padding = 'same', strides=1, name=name+'conv44_0')(uconv44)
        uconv44 = batch_Norm_Activation(uconv44, BN=self.is_bn, name=name+'bn44_1')
        uconv44 = Conv3D(filters*2, 3, activation = self.activation, padding = 'same', strides=1, name=name+'conv44_1')(uconv44)
        shortcut8 = Conv3D(filters*2, 3, padding='same', strides=1, name=name+'conv44_2')(uconv4)
        shortcut8 = batch_Norm_Activation(shortcut8, BN=self.is_bn, name=name+'bn44_2')
        output9 = add([uconv44,shortcut8])
        
        if not isinstance(self.num_of_output, list):
            if self.num_of_output == 1:
                output = Conv3D(self.num_of_output, 1, padding="same", activation='sigmoid', name=name+'conv_f_0')(output9)
            else:
                output = Conv3D(self.num_of_output, 1, padding="same", activation='softmax', name=name+'conv_f_0')(output9)
                if is_distance:
                    distance = distance_transform(output)
                    output = [output] + distance

        else:
            output_2 = Conv3D(self.num_of_output[0], 1, activation = 'softmax', padding='same', name=name+'conv_f_0')(output9)
            output_27 = Conv3D(self.num_of_output[1], 3, activation = 'sigmoid', padding='same', name=name+'conv_f_1')(output9)          
            output = [output_2, output_27]
            if is_distance:
                distance = distance_transform(output_2)
                output = output + distance
                
        if self.deep_supervision:
            output_conv7 = Conv3D(1, 1, activation = 'sigmoid', padding = 'same', name=name+'conv7_deep')(UpSampling3D(size=(4,4,4))(output7))
            output_conv8 = Conv3D(1, 1, activation = 'sigmoid', padding = 'same', name=name+'conv8_deep')(UpSampling3D(size=(2,2,2))(output8))
            output = output+[output_conv8, output_conv7]
            
        if is_mid_feat_out:
            mid_feature = [output1, output2, output3, output4, output5]
            return output, mid_feature
        else:
            return output

