#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import backend as K
from keras.regularizers import Regularizer
from keras.constraints import Constraint

class ConstInterp(Constraint):
	def __call__(self, w):
		w *= K.cast(K.greater_equal(w, 0.), K.floatx())
		w /= K.epsilon() + K.sum(w, axis=[0,1,2], keepdims=True)
		return w

class ConstNorm(Constraint):
	def __init__(self, axis=2, nb_pixels=1):
		self.axis = axis
		self.nb_pixels=nb_pixels

	def __call__(self, w):
		w /= K.epsilon() + K.sqrt(K.sum(K.square(w), axis=self.axis, keepdims=True))
		w /= self.nb_pixels
		return w

	def get_config(self):
		return {'axis': self.axis, 'nb_pixels':self.nb_pixels}

class L1L2mean(Regularizer):
	def __init__(self, l1=0., l2=0.):
		self.l1 = l1
		self.l2 = l2

	def __call__(self, x):
		regularization = 0.
		if self.l1:
			regularization += self.l1 * K.mean(K.abs(x))
		if self.l2:
			regularization += self.l2 * K.mean(K.square(x))
		return regularization

	def get_config(self):
		return {\
		'l1': float(self.l1),\
		'l2': float(self.l2),\
		}

