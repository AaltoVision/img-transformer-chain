#!/usr/bin/python

#    Main script to execute. Creates Predictions.csv file.
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
import glob

os.chdir('General_age_estimation')

# BIG ONE:

allpointerfiles = glob.glob("../img_pointers_*")

print("Found {} pointers.".format(len(allpointerfiles)))
#programPause = raw_input("Press the <ENTER> key to continue...")

for i in range(len(allpointerfiles)):
    os.system('./PREDICT_AGE_FINAL.py -pf ../img_pointers_{} -in ../img_names_{} -w list_of_models.txt -rf result_general.txt -lf log_general.txt'.format(i+1,i+1))

# Uncomment to include the kids:
#os.chdir('../Kids_age_estimation')
#os.system('./PREDICT_AGE_FINAL.py -pf ../test_image_pointers.txt -in ../image_list.txt -w list_of_models.txt -rf result_kids.txt -lf log_kids.txt')
#os.chdir('..')
#os.system('./combine_general_and_kids_estimations.py -op General_age_estimation/result_general.txt -kp Kids_age_estimation/result_kids.txt -rp Predictions.csv')
