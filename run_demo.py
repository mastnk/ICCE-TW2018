#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.datasets import cifar10, cifar100
from keras.utils import np_utils
from keras import backend as K
from keras.metrics import categorical_accuracy

K.set_image_data_format('channels_last')

import numpy as np
from padding import PaddingReplicate2D
from pixelshuffle import PixelShuffle_calc, PixelShuffle_shape
import build_model
import imageshuffle
from BlockScramble import BlockScramble



def load_cifar10():
	(X_train, y_train), (X_test, y_test) = cifar10.load_data()
	Y_train = np_utils.to_categorical(y_train, 10)
	Y_train = np.single( Y_train )
	Y_test  = np_utils.to_categorical(y_test, 10)
	Y_test = np.single( Y_test )
	
	X_train = X_train.astype('float32')/255.0
	X_test = X_test.astype('float32')/255.0

	return (X_train, Y_train), (X_test, Y_test)

def load_cifar100():
	(X_train, y_train), (X_test, y_test) = cifar100.load_data()
	Y_train = np_utils.to_categorical(y_train, 100)
	Y_train = np.single( Y_train )
	Y_test  = np_utils.to_categorical(y_test, 100)
	Y_test = np.single( Y_test )
	
	X_train = X_train.astype('float32')/255.0
	X_test = X_test.astype('float32')/255.0

	return (X_train, Y_train), (X_test, Y_test)

modelnames = ['direct', 'catmap', 'block', 'propose']
for nb_classes in [10,100]:
	
	if( nb_classes == 10 ):
		(X_train, Y_train), (X_test0, Y_test) = load_cifar10()
	elif( nb_classes == 100 ):
		(X_train, Y_train), (X_test0, Y_test) = load_cifar100()
	
	for modelname in modelnames:
		name = modelname + '{n:03d}'.format(n=nb_classes)
		model = build_model.resnet(input_shape=[32,32,3], nb_classes=nb_classes)
		model.load_weights( 'weights/' + name + '.hdf5' )
		
		X_test = np.copy(X_test0)
		if( modelname == 'catmap' ):
			imShuffle = imageshuffle.CatMapComb(1234)
			for i in range(X_test.shape[0]):
				X_test[i,:,:,:] = imShuffle.enc( X_test[i,:,:,:] )
		
		elif( modelname == 'block' ):
			imShuffle = imageshuffle.RandBlock(1234, (4,4) )
			for i in range(X_test.shape[0]):
				X_test[i,:,:,:] = imShuffle.enc( X_test[i,:,:,:] )

		elif( modelname == 'propose' ):
			bs = BlockScramble( 'key.pkl' )
			X_test = bs.Scramble( X_test )
		
		Y_pred = model.predict( X_test )
		acc = categorical_accuracy( Y_test, Y_pred )
		acc = K.get_value(acc)
		print( '{name:s}: {acc:5.3f}'.format(name=name, acc=np.mean(acc)) )


