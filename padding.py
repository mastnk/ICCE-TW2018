#!/usr/bin/env python
# -*- coding: utf-8 -*-


import keras.backend as K
from keras.engine.topology import Layer
from keras.engine import InputSpec
from keras.utils import conv_utils

class PaddingReplicate2D(Layer):
	def __init__(self, size=(1, 1), data_format=None, **kwargs):
		super(PaddingReplicate2D, self).__init__(**kwargs)
		self.data_format = conv_utils.normalize_data_format(data_format)
		self.size = conv_utils.normalize_tuple(size, 2, 'size')
		self.input_spec = InputSpec(ndim=4)
	
	def compute_output_shape(self, input_shape):
		if( self.data_format == 'channels_first' ):
			height = self.size[0] * 2 + input_shape[2] if input_shape[2] is not None else None
			width = self.size[1] * 2 + input_shape[3] if input_shape[3] is not None else None
			return (input_shape[0], input_shape[1], height, width)
			
		elif( self.data_format == 'channels_last' ):
			height = self.size[0] * 2 + input_shape[1] if input_shape[1] is not None else None
			width = self.size[1] * 2 + input_shape[2] if input_shape[2] is not None else None
			return (input_shape[0], height, width, input_shape[3])

	def call(self, inputs):
		if( self.data_format == 'channels_first' ):
			t0 = inputs[:,:,:1,:]
			t1 = inputs[:,:,-1:,:]
			if( self.size[0] > 1 ):
				t0 = K.repeat_elements( t0, self.size[0], axis=2 )
				t1 = K.repeat_elements( t1, self.size[0], axis=2 )
			t = K.concatenate( [t0, inputs, t1], axis=2 )
			
			t0 = t[:,:,:,:1]
			t1 = t[:,:,:,-1:]
			if( self.size[1] > 1 ):
				t0 = K.repeat_elements( t0, self.size[1], axis=3 )
				t1 = K.repeat_elements( t1, self.size[1], axis=3 )
			outputs = K.concatenate( [t0, t, t1], axis=3 )
			
		elif( self.data_format == 'channels_last' ):
			t0 = inputs[:,:1,:,:]
			t1 = inputs[:,-1:,:,:]
			if( self.size[0] > 1 ):
				t0 = K.repeat_elements( t0, self.size[0], axis=1 )
				t1 = K.repeat_elements( t1, self.size[0], axis=1 )
			t = K.concatenate( [t0, inputs, t1], axis=1 )
			
			t0 = t[:,:,:1,:]
			t1 = t[:,:,-1:,:]
			if( self.size[1] > 1 ):
				t0 = K.repeat_elements( t0, self.size[1], axis=2 )
				t1 = K.repeat_elements( t1, self.size[1], axis=2 )
			outputs = K.concatenate( [t0, t, t1], axis=2 )
		
		else:
			outputs = None
		
		return outputs


	def get_config(self):
		config = {'size': self.size, 'data_format': self.data_format}
		base_config = super(PaddingReplicate2D, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

if( __name__ == '__main__' ):
	t = K.ones((4, 3))
	print(K.eval(t))

	t0 = t[:1,:] - 1
	print(K.eval(t0))

	t1 = t[-1:,:] + 1
	t1 = K.repeat_elements( t1, 3, axis=0 )
	print(K.eval(t1))
	
	t = K.concatenate( [t0, t, t1], axis=0 )
	print(K.eval(t))
