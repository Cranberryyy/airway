import tensorflow as tf
from tensorflow.keras.layers  import Conv3D, MaxPooling3D,  Add, \
            BatchNormalization, Activation, UpSampling3D, Concatenate, Dropout, Conv3DTranspose

def conv_block(input_mat,num_filters,kernel_size,batch_norm):
  X = Conv3D(num_filters,kernel_size=(kernel_size,kernel_size,kernel_size),strides=(1,1,1),padding='same')(input_mat)
  if batch_norm:
    X = BatchNormalization()(X)
  
  X = Activation('relu')(X)

  X = Conv3D(num_filters,kernel_size=(kernel_size,kernel_size,kernel_size),strides=(1,1,1),padding='same')(X)
  if batch_norm:
    X = BatchNormalization()(X)
  
  X = Activation('relu')(X)
    
  X = Add()([input_mat,X])
  
  return X

class VNet():
    def __init__(self, init_channel=8, num_of_output=2, activation='relu',
                  dropout = 0.2, batch_norm = True, deep_supervision=False):
        self.init_channel = init_channel
        self.num_of_output = num_of_output
        self.deep_supervision = deep_supervision
        self.activation = activation
        self.dropout = dropout
    
    def __call__(self, x, name='vnet', is_mid_feat_out=False):
        n_filters = self.init_channel
        dropout = self.dropout

        #c1 = conv_block(input_img,n_filters,3,batch_norm)
        c1 = Conv3D(n_filters,kernel_size = (5,5,5) , strides = (1,1,1) , padding='same')(x)
        #c1 = add([c1,input_img])
        
        c2 = Conv3D(n_filters*2,kernel_size = (2,2,2) , strides = (2,2,2) , padding = 'same' )(c1)
        
        c3 = conv_block(c2 , n_filters*2,5,True)
        
        p3 = Conv3D(n_filters*4,kernel_size = (2,2,2) , strides = (2,2,2), padding = 'same')(c3)
        p3 = Dropout(dropout)(p3)
        
        c4 = conv_block(p3, n_filters*4,5,True)
        p4 = Conv3D(n_filters*8,kernel_size = (2,2,2) , strides = (2,2,2) , padding='same')(c4)
        p4 = Dropout(dropout)(p4)
            
        c5 = conv_block(p4, n_filters*8,5,True)
        p6 = Conv3D(n_filters*16,kernel_size = (2,2,2) , strides = (2,2,2) , padding='same')(c5)
        p6 = Dropout(dropout)(p6)
        #c6 = conv_block(p5, n_filters*8,5,True)
        #p6 = Conv3D(n_filters*16,kernel_size = (2,2,2) , strides = (2,2,2) , padding='same')(c6)

        p7 = conv_block(p6,n_filters*16,5,True)
            
        u6 = Conv3DTranspose(n_filters*8, (2,2,2), strides=(2, 2, 2), padding='same')(p7);
        u6 = Concatenate()([u6,c5])
        c7 = conv_block(u6,n_filters*16,5,True)
        c7 = Dropout(dropout)(c7)
        u7 = Conv3DTranspose(n_filters*4,(2,2,2),strides = (2,2,2) , padding= 'same')(c7);

        
        u8 = Concatenate()([u7,c4])
        c8 = conv_block(u8,n_filters*8,5,True)
        c8 = Dropout(dropout)(c8)
        u9 = Conv3DTranspose(n_filters*2,(2,2,2),strides = (2,2,2) , padding= 'same')(c8);
            
        u9 = Concatenate()([u9,c3])
        c9 = conv_block(u9,n_filters*4,5,True)
        c9 = Dropout(dropout)(c9)
        u10 = Conv3DTranspose(n_filters,(2,2,2),strides = (2,2,2) , padding= 'same')(c9);
        
        
        u10 = Concatenate()([u10,c1])
        c10 = Conv3D(n_filters*2,kernel_size = (5,5,5),strides = (1,1,1) , padding = 'same')(u10);
        c10 = Dropout(dropout)(c10)
        c10 = Add()([c10,u10])
        

        if not isinstance(self.num_of_output, list):
            output = [Conv3D(self.num_of_output, 1, activation = 'sigmoid', padding = 'same', name=name+'conv10_1')(c10)]
        else:
            output_2 = Conv3D(self.num_of_output[0], 3, activation = 'softmax', padding = 'same', name=name+'conv10_1_0')(c10)

            output_27 = Conv3D(self.num_of_output[1], 3, activation = 'softmax', padding = 'same', name=name+'conv10_1_1')(soft_argmax)          
            output = [output_2, output_27]

        if self.deep_supervision:
            # import pdb; pdb.set_trace()
            output_conv7 = Conv3D(self.num_of_output, 3, activation = 'sigmoid', padding = 'same', name=name+'conv7_deep')(UpSampling3D(size=(4,4,4))(c9))
            output_conv8 = Conv3D(self.num_of_output, 3, activation = 'sigmoid', padding = 'same', name=name+'conv8_deep')(UpSampling3D(size=(2,2,2))(c8))
            output = output+[output_conv8, output_conv7]
        
        if is_mid_feat_out:
            mid_feature = [conv1, conv2, conv3, conv4, conv5]
            return output, mid_feature
        else:
            return output