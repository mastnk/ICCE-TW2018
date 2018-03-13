#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import backend as K
from keras.layers import Layer, Lambda, Conv2D
from keras.initializers import RandomUniform

from regconst import ConstInterp
from padding import PaddingReplicate2D

def PixelShuffle_calc(inputs, r):
	input_shape = K.int_shape(inputs)
	batch_size, h, w, c = input_shape
	if batch_size is None:
		batch_size = -1
	rh = r
	rw = r
	oh, ow = h * rh, w * rw
	oc = c // (rh * rw)

	outputs = K.reshape(inputs, (batch_size, h, w, rh, rw, oc))
	outputs = K.permute_dimensions(outputs, (0, 1, 3, 2, 4, 5))
	outputs = K.reshape(outputs, (batch_size, oh, ow, oc))
	return outputs

def PixelShuffle_shape(s, r):
	h = s[1] * r
	w = s[2] * r
	c = s[3] // (r*r)
	return (s[0], h, w, c)

def PixelShuffle( x, ratio ):
	return Lambda( lambda x: PixelShuffle_calc(x, ratio), lambda s: PixelShuffle_shape(s, ratio) ) (x)

def PSUpsampling( x, nb_features, ratio ):
	p1 = PaddingReplicate2D((1,1))
	x = Conv2D( nb_features*ratio*ratio, (3,3), activation='linear', padding='valid', kernel_initializer=RandomUniform(0., 1.0), kernel_constraint=ConstInterp(), use_bias=False ) (p1(x))
	x = PixelShuffle( x, ratio )
	x = Conv2D( nb_features, (3,3), activation='linear', padding='valid', kernel_initializer=RandomUniform(0., 1.0), kernel_constraint=ConstInterp(), use_bias=False ) (p1(x))
	return x
