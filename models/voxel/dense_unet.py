import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, MaxPooling3D, \
        concatenate, BatchNormalization, Activation

def DenseBlock(channels, inputs, activation='relu'):

    conv1_1 = Conv3D(channels, 1,activation=None, padding='same')(inputs)
    conv1_1=BatchActivate(conv1_1, activation)
    conv1_2 = Conv3D(channels//4, 3, activation=None, padding='same')(conv1_1)
    conv1_2 = BatchActivate(conv1_2, activation)

    conv2=concatenate([inputs,conv1_2])
    conv2_1 = Conv3D(channels, 1, activation=None, padding='same')(conv2)
    conv2_1 = BatchActivate(conv2_1, activation)
    conv2_2 = Conv3D(channels // 4, 3, activation=None, padding='same')(conv2_1)
    conv2_2 = BatchActivate(conv2_2, activation)

    conv3 = concatenate([inputs, conv1_2,conv2_2])
    conv3_1 = Conv3D(channels, 1, activation=None, padding='same')(conv3)
    conv3_1 = BatchActivate(conv3_1, activation)
    conv3_2 = Conv3D(channels // 4, 3, activation=None, padding='same')(conv3_1)
    conv3_2 = BatchActivate(conv3_2, activation)

    conv4 = concatenate([inputs, conv1_2, conv2_2,conv3_2])
    conv4_1 = Conv3D(channels, 1, activation=None, padding='same')(conv4)
    conv4_1 = BatchActivate(conv4_1, activation)
    conv4_2 = Conv3D(channels // 4, 3, activation=None, padding='same')(conv4_1)
    conv4_2 = BatchActivate(conv4_2, activation)
    result=concatenate([inputs,conv1_2, conv2_2,conv3_2,conv4_2])
    return result

def BatchActivate(x, activation='relu'):
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x


class DenseUNet():
    def __init__(self, init_channel=8, num_of_output=2, activation='relu', deep_supervision=False):
        self.init_channel = init_channel
        self.num_of_output = num_of_output
        self.deep_supervision = deep_supervision
        self.activation = activation
    
    def __call__(self, x, name='denseunet3d'):
        filters=self.init_channel
        keep_prob=0.9
        block_size=7
        conv1 = Conv3D(filters * 1, 3, activation=None, padding="same")(x)
        conv1 = BatchActivate(conv1, self.activation)
        conv1 = DenseBlock(filters * 1, conv1, self.activation)
        pool1 = MaxPooling3D((2, 2, 2))(conv1)

        conv2 = DenseBlock(filters * 2, pool1, self.activation)
        pool2 = MaxPooling3D((2, 2, 2))(conv2)

        conv3 = DenseBlock(filters * 4, pool2, self.activation)
        pool3 = MaxPooling3D((2, 2, 2))(conv3)

        convm = DenseBlock(filters * 8, pool3, self.activation)

        deconv3 = Conv3DTranspose(filters * 4, 3, strides=2, padding="same")(convm)
        uconv3 = concatenate([deconv3, conv3])
        uconv3 = Conv3D(filters * 4, 1, activation=None, padding="same")(uconv3)
        uconv3 = BatchActivate(uconv3, self.activation)
        uconv3 = DenseBlock(filters * 4, uconv3, self.activation)


        deconv2 = Conv3DTranspose(filters * 2, 3, strides=2, padding="same")(uconv3)
        uconv2 = concatenate([deconv2, conv2])
        uconv2 = Conv3D(filters * 2, 1, activation=None, padding="same")(uconv2)
        uconv2 = BatchActivate(uconv2, self.activation)
        uconv2 = DenseBlock(filters * 2, uconv2, self.activation)

        deconv1 = Conv3DTranspose(filters * 1, 3, strides=2, padding="same")(uconv2)
        uconv1 = concatenate([deconv1, conv1])
        uconv1 = Conv3D(filters * 1, 1, activation=None, padding="same")(uconv1)
        uconv1 = BatchActivate(uconv1, self.activation)
        uconv1 = DenseBlock(filters * 1, uconv1, self.activation)

        if not isinstance(self.num_of_output, list):
            output = Conv3D(self.num_of_output, 1, activation = 'softmax', name=name+'conv10_1')(uconv1)
        else:
            output_2 = Conv3D(self.num_of_output[0], 1, activation = 'softmax', name=name+'conv10_1_0')(uconv1)
            soft_argmax = tf.nn.softmax(output_2, axis=-1)
            output_27 = Conv3D(self.num_of_output[1], 3, activation = 'softmax', name=name+'conv10_1_1')(soft_argmax)          
            output = [output_2, output_27]
        return output

if __name__ == '__main__':
  model = DenseUNet(nClasses = 1)
  model.summary()