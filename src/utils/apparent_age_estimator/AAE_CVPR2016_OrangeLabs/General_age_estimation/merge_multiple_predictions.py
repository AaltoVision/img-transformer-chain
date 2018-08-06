#!/usr/bin/python

#    Merges predictions by several CNNs.
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

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-lpf', action = 'store', dest='list_of_prediction_files', required=True, help='Prediction files to merge.')
parser.add_argument('-il', action = 'store', dest='list_of_images', required=True, help='List of images.')
parser.add_argument('-rf', action = 'store', dest='resulting_file', required=True, help='Resulting file in submission format.')
parser.add_argument('-rp', action = 'store', dest='resulting_probabilities', required=True, help='File with merged predictions.')
parser.add_argument('-addlog', action = 'store', dest='addlog', type=int, default=0, required=False, help='Do we append to results?')

command_args = parser.parse_args()

def calculate_age(age_vector):
	if (np.sum(age_vector) <= 0):
		age_vector = np.ones((100), dtype = np.float32)
	age_vector /= np.sum(age_vector)
	all_ages = np.asarray(range(100))
	return np.dot(age_vector, all_ages)

all_data = []
for prediction_file in command_args.list_of_prediction_files.split(' '):
	with open(prediction_file, 'r') as data_file:
		all_data += [data_file.readlines()]
with open(command_args.list_of_images, 'r') as list_of_images:
	image_list = list_of_images.readlines()
	image_list = map(lambda x: x.split('\n')[0], image_list)

print("all_data / image_list.shape:")
print(np.shape(all_data))
print(np.shape(image_list))

averaged_predictions = np.zeros((len(image_list), 100), dtype = np.float32)
for data_ind in range(len(all_data)):
	for img_ind in range(len(image_list)):
		averaged_predictions[img_ind, :] += np.array(map(lambda x: float(x.split('\n')[0]), all_data[data_ind][img_ind].split(' ')))

if command_args.addlog == 1:
	print("*Adding* to results at {}".format(command_args.resulting_file))
else:
	print("*Creating* results at {}".format(command_args.resulting_file))

with open(command_args.resulting_file, 'a') as resulting_file:
	for img_ind in range(len(image_list)):
		resulting_file.write(image_list[img_ind] + ',' + str(calculate_age(averaged_predictions[img_ind, :])) + '\n')

with open(command_args.resulting_probabilities, 'w') as resulting_probabilities:
	for img_ind in range(len(image_list)):
		for age_ind in range(100):
			if age_ind <= 98:
				resulting_probabilities.write(str(averaged_predictions[img_ind][age_ind]) + ' ')
			else:
				resulting_probabilities.write(str(averaged_predictions[img_ind][age_ind]) + '\n')

