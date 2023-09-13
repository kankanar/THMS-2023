"""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/11
"""

#import keras,os
from keras.models import Model
#from keras.layers import add,multiply
#from keras.layers import Lambda,Input, Conv2D,Conv2DTranspose,Conv2DTranspose, MaxPooling2D, UpSampling2D,Cropping2D, core, Dropout,normalization,concatenate,Activation
from keras import backend as K
#rom keras.layers.core import Layer, InputSpec
#from keras.layers.advanced_activations import LeakyReLU
#from keras.utils import plot_model
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
#import numpy as np 
#import os
#import skimage.io as io
#import skimage.transform as trans
from keras.models import *
from keras.layers import *
#from keras import backend as keras
from losses import (
    binary_crossentropy,
    dice_loss,
    bce_dice_loss,
    dice_coef,
    weighted_bce_dice_loss,
    dice_coef_loss,
    focal_loss_fixed,
    focal_loss,
    bce_dice_coef_loss,
    focal_tversky
)


def DenseBlock(inputs, outdim):

        inputshape = K.int_shape(inputs)
        bn = BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
                                              beta_initializer='zero', gamma_initializer='one')(inputs)
        act = Activation('relu')(bn)
        conv1 = Conv2D(outdim, (3, 3), activation=None, padding='same')(act)

        if inputshape[3] != outdim:
            shortcut = Conv2D(outdim, (1, 1), padding='same')(inputs)
        else:
            shortcut = inputs
        result1 = add([conv1, shortcut])

        bn = BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
                                              beta_initializer='zero', gamma_initializer='one')(result1)
        act = Activation('relu')(bn)
        conv2 = Conv2D(outdim, (3, 3), activation=None, padding='same')(act)
        result = add([result1, conv2, shortcut])
        result = Activation('relu')(result)
        return result
def bottleneck(x, filters_bottleneck, mode='cascade', depth=6,
               kernel_size=(3, 3), activation='relu'):
    dilated_layers = []
    if mode == 'cascade':  # used in the competition
        for i in range(depth):
            x = Conv2D(filters_bottleneck, kernel_size,
                       activation=activation, padding='same', dilation_rate=2**i)(x)
            dilated_layers.append(x)
        return add(dilated_layers)
    elif mode == 'parallel':  # Like "Atrous Spatial Pyramid Pooling"
        for i in range(depth):
            dilated_layers.append(
                Conv2D(filters_bottleneck, kernel_size,
                       activation=activation, padding='same', dilation_rate=2**i)(x)
            )
        return add(dilated_layers)

def dense_unet(pretrained_weights = None,input_size = (384,384,3)):
        inputs = Input(input_size)
        conv1 = Conv2D(64, (1, 1), activation=None, padding='same')(inputs)
        conv1 = BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
                                                 beta_initializer='zero', gamma_initializer='one')(conv1)
        conv1 = Activation('relu')(conv1)

        conv1 = DenseBlock(conv1, 128)  # 256
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 =DenseBlock(pool1, 128)  # 128
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = DenseBlock(pool2, 256)  # 64
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = DenseBlock(pool3, 512)  # 32
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        
        conv5 = DenseBlock(pool4, 512)  # 32
        pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)         
        
        conv6 =  bottleneck(pool5, filters_bottleneck=32 * 2**3, mode='cascade')        # 16

        up1 = Conv2DTranspose(512, (3, 3), strides=2, activation='relu', padding='same')(conv6) #32
        up1 = concatenate([up1, conv5], axis=3)

        conv7 = DenseBlock(up1, 512)

        up2 = Conv2DTranspose(256, (3, 3), strides=2, activation='relu', padding='same')(conv7) #64
        up2 = concatenate([up2, conv4], axis=3)

        conv8 = DenseBlock(up2, 256)

        up3 = Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(conv8)#128
        up3 = concatenate([up3, conv3], axis=3)
        
        conv9 = DenseBlock(up3, 128)

        up4 = Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(conv9)#256
        up4 = concatenate([up4, conv2], axis=3)

        conv10 = DenseBlock(up4, 64)
        
        up5 = Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(conv10)#256
        up4 = concatenate([up5, conv1], axis=3)
        
        conv11 = DenseBlock(up5, 64)
        
        act = Conv2D(1, 1, activation = 'sigmoid')(conv11)
        
        model = Model(inputs=inputs, outputs=act)
        model.compile(optimizer = Adam(lr = 1e-4), loss = ['binary_crossentropy'], metrics = ['accuracy'])
		#plot_model(model, to_file=os.path.join(self.config.checkpoint, "model.png"), show_shapes=True)
        return model