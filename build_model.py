#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Input, Lambda
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.merge import Add, Concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.datasets import cifar10, cifar100
from keras.utils import np_utils
import keras.backend as K

from padding import PaddingReplicate2D
from pixelshuffle import PixelShuffle

import numpy as np
import random

def resnet(input_shape, nb_classes=10):
	p1 = PaddingReplicate2D((1,1))
	inp = Input(shape=input_shape)
	x = inp

	# block0 ####################################################
	x = BatchNormalization() (x)
	x = Conv2D( 4*4*3, (4,4), padding='valid', strides=(4,4), kernel_initializer='he_normal' ) (x)
	x = Activation('relu') (x)
	# 8x8x48
	
	x = BatchNormalization() (x)
	x = Conv2D( 4*4*3, (1,1), padding='valid', kernel_initializer='he_normal' ) (x)
	x = Activation('relu') (x)
	# 8x8x48

	x = BatchNormalization() (x)
	x = Conv2D( 4*4*32, (1,1), padding='valid', kernel_initializer='he_normal' ) (x)
	x = PixelShuffle( x, 4 )
	# 32x32x32


	# block1 ####################################################
	f = 32
	fs = 16
	for i in range(2):
		f0 = f
		f = f+fs
		r = x
		
		r = BatchNormalization() (r)
		r = Activation('relu') (r)
		r = Conv2D( f, (3,3), padding='valid', kernel_initializer='he_normal' ) (p1(r))

		r = BatchNormalization() (r)
		r = Activation('relu') (r)
		r = Conv2D( f, (3,3), padding='valid', kernel_initializer='he_normal' ) (p1(r))
		
		r = Dropout(0.25) (r)

		r0 = Lambda(lambda x: x[:,:,:,:f0], lambda s: (s[0], s[1], s[2], f0) ) (r)
		r1 = Lambda(lambda x: x[:,:,:,f0:], lambda s: (s[0], s[1], s[2], fs) ) (r)
		
		x = Add() ( [x,r0] )
		x = Concatenate(axis=-1) ([x,r1])
		
		f0 = f
	# 32x32x64

	x = MaxPooling2D(pool_size=(2, 2)) (x)
	# 16x16x64


	# block2 ####################################################
	f = 64
	fs = 32
	for i in range(2):
		f0 = f
		f = f+fs
		r = x
		
		r = BatchNormalization() (r)
		r = Activation('relu') (r)
		r = Conv2D( f, (3,3), padding='valid', kernel_initializer='he_normal' ) (p1(r))

		r = BatchNormalization() (r)
		r = Activation('relu') (r)
		r = Conv2D( f, (3,3), padding='valid', kernel_initializer='he_normal' ) (p1(r))
		
		r = Dropout(0.25) (r)

		r0 = Lambda(lambda x: x[:,:,:,:f0], lambda s: (s[0], s[1], s[2], f0) ) (r)
		r1 = Lambda(lambda x: x[:,:,:,f0:], lambda s: (s[0], s[1], s[2], fs) ) (r)

		x = Add() ( [x,r0] )
		x = Concatenate(axis=-1) ([x,r1])
		
		f0 = f
	# 16x16x128
	
	x = MaxPooling2D(pool_size=(2, 2)) (x)
	# 8x8x128

	# block3 ####################################################
	f = 128
	fs = 64
	for i in range(2):
		f0 = f
		f = f+fs
		r = x
		
		r = BatchNormalization() (r)
		r = Activation('relu') (r)
		r = Conv2D( f, (3,3), padding='valid', kernel_initializer='he_normal' ) (p1(r))

		r = BatchNormalization() (r)
		r = Activation('relu') (r)
		r = Conv2D( f, (3,3), padding='valid', kernel_initializer='he_normal' ) (p1(r))
		
		r = Dropout(0.25) (r)

		r0 = Lambda(lambda x: x[:,:,:,:f0], lambda s: (s[0], s[1], s[2], f0) ) (r)
		r1 = Lambda(lambda x: x[:,:,:,f0:], lambda s: (s[0], s[1], s[2], fs) ) (r)

		x = Add() ( [x,r0] )
		x = Concatenate(axis=-1) ([x,r1])
		
		f0 = f
	# 8x8x256
	
	x = MaxPooling2D(pool_size=(2, 2)) (x)
	# 4x4x256

	# block4 ####################################################
	x = Dropout(0.5) (x)
	x = Conv2D( 512, (1,1), padding='valid', kernel_initializer='he_normal', activation='relu' ) (x)

	x = Dropout(0.5) (x)
	x = Conv2D( nb_classes, (1,1), padding='valid', kernel_initializer='he_normal' ) (x)
	x = GlobalAveragePooling2D() (x)
	x = Activation('softmax') (x)

	return Model(inputs=inp, outputs=x)

