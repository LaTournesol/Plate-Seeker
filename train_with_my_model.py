import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model

import argparse
import os
import sys

class_num = 37
parser = argparse.ArgumentParser(description='Create custom resnet model and train the model on training data')
parser.add_argument(
	'-d',
	'--data_path',
	help='path to the .npz data file that contains the test images and their ground truth labels',
	default='large_training_set.npz')
parser.add_argument(
	'-l',
	'--load_weights',
	help='whether or not (y/n) to load previously trained weights',
	default='n')
parser.add_argument(
	'-w',
	'--weight_path',
	help='path to the .h5 weight file',
	default=None)


def model(input_shape):
	# Define the input placeholder as a tensor with shape input_shape

	X_input = Input(input_shape)

	# Conv -> BN -> RELU
	X = Conv2D(16, (5, 5), strides = (1, 1), name = 'conv0')(X_input)
	X = BatchNormalization(axis = 3, name = 'bn0')(X)
	X = Activation('relu')(X)

	# MAXPOOL
	X = MaxPooling2D((2, 2), name='max_pool')(X)

	X = Flatten()(X)
	X = Dense(class_num, activation = 'softmax', name = 'fc')(X)

	model = Model(inputs = X_input, outputs = X, name = 'my_model')

	return model

def _main(args):
	data_path = os.path.expanduser(args.data_path)
	load = os.path.expanduser(args.load_weights)
	if load == 'y':
		checkpoint = os.path.expanduser(args.weight_path)
		if checkpoint is None:
			sys.exit('Weight path is needed when selected to load weights')

	number_of_iterations = 20
	number_of_epochs = 5

	data = np.load(data_path)
	X_train, Y_train, X_eval, Y_eval = data['x_train'], data['y_train'], data['x_eval'], data['y_eval']

	print('X_train has shape {}'.format(X_train.shape))
	print('Y_train has shape {}'.format(Y_train.shape))
	print('X_eval has shape {}'.format(X_eval.shape))
	print('Y_eval has shape {}'.format(Y_eval.shape))

	my_model = model((40, 40, 3))
	if load == 'y':
		my_model.load_weights(checkpoint)
	my_model.compile(optimizer='adam', loss='categorical_crossentropy')

	my_model.fit(x = X_train, y = Y_train, batch_size = 64, epochs = 30, validation_data = (X_eval, Y_eval), shuffle = True)
	my_model.save_weights('weights_epoch{}.h5'.format(30))

if __name__ == '__main__':
	_main(parser.parse_args())






