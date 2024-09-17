
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv3D, MaxPooling3D

"""
Create connectivity 
Convert connectivity to binary prediction
"""
#### create connectivity
kernel_ = np.zeros([3, 3, 3, 1, 27])  #H W D
c = 0
for i in range(3): #H
    for j in range(3): #D
        for k in range(3): #W
            kernel_[i,j,k,0,c] = 1
            kernel_[1,1,1,0,c] = 1
            c += 1
kernel_ = tf.keras.initializers.Constant(kernel_)

def create_connectivity(input, name=None):
    if name is None:
        name = 'create_connectivity'
    output = Conv3D(27, 3, use_bias=False, padding = 'SAME', kernel_initializer=kernel_, trainable=False, name=name)(input)
    output = output / 2
    return output

#### inverse connectivity
kernel = np.zeros([3, 3, 3, 27, 27])
c = 26
for i in range(3):
    for j in range(3):
        for k in range(3):
            kernel[i, j, k, c, c]  = 1
            kernel[1,1,1, c,c] =1
            c -= 1
kernel = tf.keras.initializers.Constant(kernel)

def inverse_connectivity(input_, name=None):
    if name is None:
        name = 'inverse_connectivity'
    output = Conv3D(27, 3, use_bias=False, padding = 'SAME', kernel_initializer=kernel, trainable=False, name=name)(input_)
    output = output / 2
    return output


def gumbel_softmax(logits, temperature=1.0):
    gumbel_noise = -tf.math.log(-tf.math.log(tf.random.uniform(shape=tf.shape(logits))))
    logits_with_noise = (logits + gumbel_noise) / temperature
    softmax = tf.nn.softmax(logits_with_noise, axis=-1)
    return softmax

def gumbel_argmax(logits, temperature=1.0):
    softmax = tf.exp(logits * 10)
    indices = tf.range(logits.shape[-1], dtype=tf.float32)
    argmax = tf.round(tf.reduce_sum(softmax/tf.reduce_sum(softmax, axis=-1, keepdims=True) * indices, axis=-1, keepdims=True))
    return argmax

def soft_dilate(img):
    return MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same', data_format=None)(img)

def distance_transform(label):
    label = gumbel_argmax(label)
    label_dilate = soft_dilate(label)
    boundary = label_dilate - label
    delta = label * 0
    for i in range(1, 10, 1):
        boundary_dilate = soft_dilate(boundary)
        delta_1 = label * (boundary_dilate - boundary) 
        delta_2 = delta_1 * i
        boundary = boundary_dilate
        delta = delta + tf.math.minimum(delta_2 + delta, delta_2)
    outline_to_centerline = delta
    return [outline_to_centerline]

def boundary_locate(label):
    label = gumbel_argmax(label)
    label_dilate = soft_dilate(label)
    boundary = label_dilate - label
    delta = label * 0
    for i in range(1, 2, 1):
        boundary_dilate = soft_dilate(boundary)
        delta_1 = label * (boundary_dilate - boundary) 
        delta_2 = delta_1 * i
        boundary = boundary_dilate
        delta = delta + tf.math.minimum(delta_2 + delta, delta_2)
    return [delta]
