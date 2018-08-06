#!/usr/bin/python

#    Runs a CNN with given weights on a list of given images.
#
#    Copyright Orange (C) 2016. All rights reserved.
#    G. Antipov, M. Baccouche and S. A. Berrani.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import caffe
import os
from caffe import layers as L
from caffe import params as P
import time
import numpy as np
import argparse
import sys

CONV_LR_MULT = 1
FC_LR_MULT = 1

def VGG_16(data, batch_size):
	n = caffe.NetSpec()
	n.X, n.Y = L.ImageData(ntop = 2, source = data, batch_size = batch_size, transform_param = dict(mean_value = [91.73, 102.71, 129.04]))
	n.conv1_1 = L.Convolution(n.X, kernel_size = 3, pad = 1, num_output = 64, weight_filler = dict(type = 'gaussian', std = 0.01), param = dict(lr_mult = CONV_LR_MULT))
	n.relu1_1 = L.ReLU(n.conv1_1, in_place = True)
	n.conv1_2 = L.Convolution(n.relu1_1, kernel_size = 3, pad = 1, num_output = 64, weight_filler = dict(type = 'gaussian', std = 0.01), param = dict(lr_mult = CONV_LR_MULT))
	n.relu1_2 = L.ReLU(n.conv1_2, in_place = True)
	n.pool1 = L.Pooling(n.relu1_2, kernel_size = 2, stride = 2, pool = P.Pooling.MAX)
	n.conv2_1 = L.Convolution(n.pool1, kernel_size = 3, pad = 1, num_output = 128, weight_filler = dict(type = 'gaussian', std = 0.01), param = dict(lr_mult = CONV_LR_MULT))
	n.relu2_1 = L.ReLU(n.conv2_1, in_place = True)
	n.conv2_2 = L.Convolution(n.relu2_1, kernel_size = 3, pad = 1, num_output = 128, weight_filler = dict(type = 'gaussian', std = 0.01), param = dict(lr_mult = CONV_LR_MULT))
	n.relu2_2 = L.ReLU(n.conv2_2, in_place = True)
	n.pool2 = L.Pooling(n.relu2_2, kernel_size = 2, stride = 2, pool = P.Pooling.MAX)
	n.conv3_1 = L.Convolution(n.pool2, kernel_size = 3, pad = 1, num_output = 256, weight_filler = dict(type = 'gaussian', std = 0.01), param = dict(lr_mult = CONV_LR_MULT))
	n.relu3_1 = L.ReLU(n.conv3_1, in_place = True)
	n.conv3_2 = L.Convolution(n.relu3_1, kernel_size = 3, pad = 1, num_output = 256, weight_filler = dict(type = 'gaussian', std = 0.01), param = dict(lr_mult = CONV_LR_MULT))
	n.relu3_2 = L.ReLU(n.conv3_2, in_place = True)
	n.conv3_3 = L.Convolution(n.relu3_2, kernel_size = 3, pad = 1, num_output = 256, weight_filler = dict(type = 'gaussian', std = 0.01), param = dict(lr_mult = CONV_LR_MULT))
	n.relu3_3 = L.ReLU(n.conv3_3, in_place = True)
	n.pool3 = L.Pooling(n.relu3_3, kernel_size = 2, stride = 2, pool = P.Pooling.MAX)
	n.conv4_1 = L.Convolution(n.pool3, kernel_size = 3, pad = 1, num_output = 512, weight_filler = dict(type = 'gaussian', std = 0.01), param = dict(lr_mult = CONV_LR_MULT))
	n.relu4_1 = L.ReLU(n.conv4_1, in_place = True)
	n.conv4_2 = L.Convolution(n.relu4_1, kernel_size = 3, pad = 1, num_output = 512, weight_filler = dict(type = 'gaussian', std = 0.01), param = dict(lr_mult = CONV_LR_MULT))
	n.relu4_2 = L.ReLU(n.conv4_2, in_place = True)
	n.conv4_3 = L.Convolution(n.relu4_2, kernel_size = 3, pad = 1, num_output = 512, weight_filler = dict(type = 'gaussian', std = 0.01), param = dict(lr_mult = CONV_LR_MULT))
	n.relu4_3 = L.ReLU(n.conv4_3, in_place = True)
	n.pool4 = L.Pooling(n.relu4_3, kernel_size = 2, stride = 2, pool = P.Pooling.MAX)
	n.conv5_1 = L.Convolution(n.pool4, kernel_size = 3, pad = 1, num_output = 512, weight_filler = dict(type = 'gaussian', std = 0.01), param = dict(lr_mult = CONV_LR_MULT))
	n.relu5_1 = L.ReLU(n.conv5_1, in_place = True)
	n.conv5_2 = L.Convolution(n.relu5_1, kernel_size = 3, pad = 1, num_output = 512, weight_filler = dict(type = 'gaussian', std = 0.01), param = dict(lr_mult = CONV_LR_MULT))
	n.relu5_2 = L.ReLU(n.conv5_2, in_place = True)
	n.conv5_3 = L.Convolution(n.relu5_2, kernel_size = 3, pad = 1, num_output = 512, weight_filler = dict(type = 'gaussian', std = 0.01), param = dict(lr_mult = CONV_LR_MULT))
	n.relu5_3 = L.ReLU(n.conv5_3, in_place = True)
	n.pool5 = L.Pooling(n.relu5_3, kernel_size = 2, stride = 2, pool = P.Pooling.MAX)
	n.fc6 = L.InnerProduct(n.pool5, num_output = 4096, weight_filler = dict(type = 'gaussian', std = 0.01), param = dict(lr_mult = FC_LR_MULT))
	n.relu6 = L.ReLU(n.fc6, in_place = True)
	n.drop6 = L.Dropout(n.relu6, dropout_ratio = 0.5)
	n.fc7 = L.InnerProduct(n.drop6, num_output = 4096, weight_filler = dict(type = 'gaussian', std = 0.01), param = dict(lr_mult = FC_LR_MULT))
	n.relu7 = L.ReLU(n.fc7, in_place = True)
	n.drop7 = L.Dropout(n.relu7, dropout_ratio = 0.5)
	n.fc8_class_age = L.InnerProduct(n.drop7, num_output = 100, weight_filler = dict(type = 'gaussian', std = 0.01), param = dict(lr_mult = FC_LR_MULT))
	n.prediction = L.Sigmoid(n.fc8_class_age)
	n.target = L.Power(n.Y, power_param = dict(power = 1, scale = 1, shift = 0))

	return n.to_proto()

def calculate_age(age_vector):
	if (np.sum(age_vector) <= 0):
		age_vector = np.ones((100), dtype = np.float32)
	age_vector /= np.sum(age_vector)
	all_ages = np.asarray(range(100))
	return np.dot(age_vector, all_ages)

parser = argparse.ArgumentParser()
parser.add_argument('-w', action = 'store', dest='weights', required=True, help='File with weights (trained model).')
parser.add_argument('-td', action = 'store', dest='test_data', required=True, help='Pointer to the test data.')
parser.add_argument('-il', action = 'store', dest='image_list', required=True, help='Image names.')
parser.add_argument('-rf', action = 'store', dest='result_file', required=True, help='File to submit.')
parser.add_argument('-prob', action = 'store', dest='probabilities_file', required=True, help='File with resulting probabilities.')
command_args = parser.parse_args()

BATCH_SIZE = 64

with open(command_args.test_data, 'r') as test_data_file:
	test_data = test_data_file.readlines()
	test_data = map(lambda x: x.split('\n')[0] + ' 0\n', test_data)
with open(command_args.image_list, 'r') as image_list:
	list_of_images = image_list.readlines()
	list_of_images = map(lambda x: x.split('\n')[0], list_of_images)

temp_file_name = 'temp_pointer_' + str(os.getpid()) + '.txt'

print("Result file: {}".format(command_args.result_file))
curr_iter = 0

from os import listdir
from os.path import isfile, join

def get_file_list(mypath):
	return [f for f in listdir(mypath) if isfile(join(mypath, f))]

INPUT_FILES_DIR = '../test_images'

Mods = False

if Mods:
	print("list of images:")
	print(list_of_images)
	if not (INPUT_FILES_DIR is None):
		file_list = get_file_list(INPUT_FILES_DIR)
		#list_of_images = file_list
	for i in range(10):
		if (len(file_list) > i+1):
			print(file_list[i])

	print("list of images v2:")
	print(list_of_images)

	wait = raw_input('Press enter')

with open(command_args.result_file, 'w') as result_file, open(command_args.probabilities_file, 'w') as output_probabilities:
	while (curr_iter < len(list_of_images)):
		current_batch_size = min(BATCH_SIZE, len(list_of_images) - curr_iter)
		with open(temp_file_name, 'w') as temp_ptr_file:
			temp_ptr_file.writelines(test_data[curr_iter : (curr_iter + current_batch_size)])

		with open('vgg_16_test.prototxt', 'w') as f:
			f.write(str(VGG_16(temp_file_name, current_batch_size)))

		caffe.set_device(0)
		caffe.set_mode_gpu()
		start_time = time.time()
		print 'Loading the weights...'
		net = caffe.Net('vgg_16_test.prototxt', command_args.weights, caffe.TEST)
		finish_time = time.time()
		print 'The weights are loaded in {0:.1f} seconds'.format(finish_time - start_time)

		print 'Processing images [{0}, {1}] out of {2}'.format(curr_iter, curr_iter + current_batch_size - 1, len(list_of_images))

		out = net.forward()
		print out['target'][0 : current_batch_size]
		print map(lambda x: calculate_age(x), out['prediction'][:])
		for write_iter in range(current_batch_size):
			calculated_age = calculate_age(out['prediction'][write_iter])
			img_ind = out['target'][write_iter]
			print 'Predicted value for image {0} is {1}'.format(img_ind, calculated_age)
			#print 'Predicted value for image {0} is {1} at iter {2}'.format(list_of_images[curr_iter + write_iter], calculated_age, curr_iter)
			result_file.write(list_of_images[curr_iter + write_iter] + ',' + str(calculated_age) + '\n')
		for i in range(len(out['prediction'])):
			for j in range(len(out['prediction'][i])):
				if j < (len(out['prediction'][i]) - 1):
					output_probabilities.write(str(out['prediction'][i][j]) + ' ')
				else:
					output_probabilities.write(str(out['prediction'][i][j]) + '\n')

		curr_iter += current_batch_size

os.system('rm ' + temp_file_name)

