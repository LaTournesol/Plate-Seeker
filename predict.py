import argparse
import os

from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model
import numpy as np
import cv2

class_num = 37
parser = argparse.ArgumentParser(description='Run the trained resNet model on test images to predict their labels')
parser.add_argument(
	'-d',
	'--data_path',
	help='path to the .npz data file that contains the test images and their ground truth labels',
	default='large_training_set.npz')
parser.add_argument(
	'-w',
	'--weight_path',
	help='path to the .h5 file that contains the trained weights of the resnet model',
	default='weights_epoch_best.h5')

def _main(args):
	data_path = os.path.expanduser(args.data_path)
	weight_path = os.path.expanduser(args.weight_path)
	assert weight_path.endswith('.h5'), 'You need to provide a .h5 file'

	data = np.load(data_path)
	my_model = model((40, 40, 3))
	my_model.load_weights(weight_path)

	X_test, Y_test = data['x_test'], data['y_test']
	Y_pred = predict(my_model, X_test)

	correct_labels, wrong_labels = compare(Y_pred, Y_test, X_test)
	print(wrong_labels)

	# visualize(Y_pred, Y_eval, X_eval, correct_labels, wrong_labels)




def create_custom_model(input_shape, num_classes):
	base_model = ResNet50(include_top = False, weights = 'imagenet', input_shape = input_shape, classes = num_classes)

	X = base_model.output
	X = Flatten()(X)

	predictions = Dense(num_classes, activation = 'softmax', input_shape = X.shape)(X)

	model = Model(inputs = base_model.input, outputs = predictions)
	return model

def process_pred_data(Y_pred):
	# change y_pred array returned by model.predict to a nicer form
	# where in each example, the highest prob will be given a 1 and 0 otherwise

	for pred in Y_pred:
		max_index = np.argmax(pred)
		pred[:max_index] = 0
		pred[max_index+1:] = 0
		pred[max_index] = 1

	return Y_pred

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

def predict(model, X_eval):
	Y_pred = process_pred_data(model.predict(X_eval, verbose=0))

	return Y_pred

def get_y_label(i):
	if i < 10:
		return str(i)
	elif i <= 17:
		return chr(i+55)
	elif i <= 22:           # skip letter I
		return chr(i+56)
	elif i <= 33:           # skip letter O
		return chr(i+57)
	elif i == 34:
		return 'su'
	elif i == 35:
		return 'zhe'
	elif i == 36:
		return 'hu'
	else:
		return 'err'
	

def compare(Y_pred, Y_eval, X_eval):
	# see how well the model is doing

	assert Y_pred.shape == Y_eval.shape, 'Prediction vector and groundtruth vector should have the same shape'

	total_num = len(Y_eval)
	correct_labels = []
	wrong_labels = []

	for i in range(total_num):
		true_label = get_y_label(np.argmax(Y_eval[i]))
		pred_label = get_y_label(np.argmax(Y_pred[i]))
		if (Y_eval[i] == Y_pred[i]).all():
			correct_labels.append((pred_label, true_label))
		else:
			wrong_labels.append((pred_label, true_label, i))

	print('Out of {} number of images, \nthe model got {} number of them correctly.'.format(total_num, len(correct_labels)))

	return correct_labels, wrong_labels

def visualize(Y_pred, Y_eval, X_eval, correct_labels, wrong_labels):
	print('correct: {}'.format(correct_labels[:10]))
	print('wrong: {}'.format(wrong_labels[:10]))

	img1 = X_eval[wrong_labels[7][2], :, :, :]
	cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)

	cv2.imshow('img', img1)
	cv2.waitKey(0)
	cv2.destroyAllWindows()





if __name__ == '__main__':
	_main(parser.parse_args())