#!/usr/bin/python

#    Executes apparent age estimation by general CNNs.
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

import os
import numpy as np
import argparse
import time
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('-pf', action = 'store', dest='pointers_file', required=True, help='File with pointers to test images.')
parser.add_argument('-in', action = 'store', dest='image_names', required=True, help='Image names to appear in the output file.')
parser.add_argument('-w', action = 'store', dest='models_file', required=True, help='File containing the list of all models.')
parser.add_argument('-rf', action = 'store', dest='resulting_file', required=True, help='File with resulting predictions.')
parser.add_argument('-lf', action = 'store', dest='log_file', required=True, help='File with logs of all actions.')

command_args = parser.parse_args()
with open(command_args.models_file, 'r') as models_file:
	list_of_models = map(lambda x: x.split('\n')[0], models_file)

DoMerge = True
RemoveAllTemps = True

with open(command_args.log_file, 'a') as log_file:
	ts = time.time()
	st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
	log_file.write(st + ': temporary image files are going to be created...\n')
	os.system('./create_multiple_copies_of_images.py -pf ' + command_args.pointers_file)
	ts = time.time()
	st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
	log_file.write(st + ': temporary image files have been created...\n')

	for model_ind in range(len(list_of_models)):
		ts = time.time()
		st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
		log_file.write(st + ': age estimations by the model ' + list_of_models[model_ind] + ' are going to be done ...\n')
		os.system('./evaluate_competition.py -w ' + list_of_models[model_ind] + ' -td temp_original.txt -il ' + command_args.image_names + ' -rf temp_out_original.txt -prob temp_prob_original.txt')
		if DoMerge:
			os.system('./evaluate_competition.py -w ' + list_of_models[model_ind] + ' -td temp_mir.txt -il ' + command_args.image_names + ' -rf temp_out_mir.txt -prob temp_prob_mir.txt')
			os.system('./evaluate_competition.py -w ' + list_of_models[model_ind] + ' -td temp_rot+.txt -il ' + command_args.image_names + ' -rf temp_out_rot+.txt -prob temp_prob_rot+.txt')
			os.system('./evaluate_competition.py -w ' + list_of_models[model_ind] + ' -td temp_rot-.txt -il ' + command_args.image_names + ' -rf temp_out_rot-.txt -prob temp_prob_rot-.txt')
			os.system('./evaluate_competition.py -w ' + list_of_models[model_ind] + ' -td temp_shift+.txt -il ' + command_args.image_names + ' -rf temp_out_shift+.txt -prob temp_prob_shift+.txt')
			os.system('./evaluate_competition.py -w ' + list_of_models[model_ind] + ' -td temp_shift-.txt -il ' + command_args.image_names + ' -rf temp_out_shift-.txt -prob temp_prob_shift-.txt')
			os.system('./evaluate_competition.py -w ' + list_of_models[model_ind] + ' -td temp_sc+.txt -il ' + command_args.image_names + ' -rf temp_out_sc+.txt -prob temp_prob_sc+.txt')
			os.system('./evaluate_competition.py -w ' + list_of_models[model_ind] + ' -td temp_sc-.txt -il ' + command_args.image_names + ' -rf temp_out_sc-.txt -prob temp_prob_sc-.txt')
			os.system('./merge_multiple_predictions.py -lpf \'temp_prob_original.txt temp_prob_mir.txt\' -il ' + command_args.image_names + ' -rf temp_unused_1.txt -rp temp_probs_1.txt')
			os.system('./merge_multiple_predictions.py -lpf \'temp_prob_rot+.txt temp_prob_rot-.txt\' -il ' + command_args.image_names + ' -rf temp_unused_2.txt -rp temp_probs_2.txt')
			os.system('./merge_multiple_predictions.py -lpf \'temp_prob_shift+.txt temp_prob_shift-.txt\' -il ' + command_args.image_names + ' -rf temp_unused_3.txt -rp temp_probs_3.txt')
			os.system('./merge_multiple_predictions.py -lpf \'temp_prob_sc+.txt temp_prob_sc-.txt\' -il ' + command_args.image_names + ' -rf temp_unused_4.txt -rp temp_probs_4.txt')
			os.system('./merge_multiple_predictions.py -lpf \'temp_probs_2.txt temp_probs_3.txt temp_probs_4.txt\' -il ' + command_args.image_names + ' -rf temp_unused_5.txt -rp temp_probs_5.txt')
			os.system('./merge_multiple_predictions.py -lpf \'temp_probs_1.txt temp_probs_5.txt\' -il ' + command_args.image_names + ' -addlog=1 -rf result_model_' + str(model_ind + 1) + '.txt -rp probs_model_' + str(model_ind + 1) + '.txt')
		ts = time.time()
		st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
		log_file.write(st + ': age estimations by the model ' + list_of_models[model_ind] + ' have been done ...\n')

	os.system('rm -r TEMP')
	if RemoveAllTemps:
		os.system('rm temp_*.txt')

	if DoMerge:
		os.system('./merge_multiple_predictions.py -lpf \'probs_model_1.txt probs_model_2.txt probs_model_3.txt probs_model_4.txt probs_model_5.txt probs_model_6.txt probs_model_7.txt probs_model_8.txt probs_model_9.txt probs_model_10.txt probs_model_11.txt\' -il ' + command_args.image_names + ' -rf ' + command_args.resulting_file + ' -rp temp_unused.txt')
	os.system('rm temp_unused.txt')

