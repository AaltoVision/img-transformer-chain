import csv
import argparse
import numpy as np
import os
from shutil import copyfile

PARSER = argparse.ArgumentParser()
PARSER.add_argument("-f", action='store', required=True, dest='fname')
PARSER.add_argument("-imgdir", action='store', required=True, dest='imgdir')
ARGS = PARSER.parse_args()

def get_ages(args):
    """ Eval the ages"""
    i = 1

    all_ages = {}

    with open(ARGS.fname, "rt", encoding='utf-8') as f:
        freader = csv.reader(f, delimiter=",")
        for row in freader:
            #print("Row {}".format(i))
            try:
                if len(row) > 1:
                    age = float(row[1])
                    #if age > 22.5:
                        #print("rm \"{}\" # age = {}".format(row[0], age))
                    all_ages[row[0].encode('utf-8')] = row[1] #.encode('utf-8')
            except:
                print("Some error. Skipping this row.")
            i += 1

    return all_ages

def eval_ages(args, summary_op):
    ages = get_ages(args)
    for name, age in ages:
        summary_op(name, age)

def get_mean(ages):
	ages.keys()
	age_vals = [float(x) for x in ages.values()]
	print("Sanity check: age at [0] is {}".format(age_vals[0]))
	print("Mean age of {0} samples is {1:0.2f} y with std = {2:0.3f}.".format(len(age_vals), np.mean(age_vals), np.std(age_vals)))

# Insert your target directory here: (Ensure it exists and has correct permissions):
maindir = None # e.g. '/data/aetuned_celeb/'

if not maindir:
    print("Please add the target directory manually into the script relocate_by_ages.py as the 'maindir' variable, and re-run the pipeline.")
    import sys
    sys.exit()

ages = get_ages(ARGS)
for name, age in ages.items():
    age = int(float(age))
    ok_dir = '{}age_{}'.format(maindir, age)
    print("Try ok_dir: {}".format(ok_dir))
    if not os.access(ok_dir, os.F_OK): #if exists
        print("(mkdir) {}".format(ok_dir))
        os.mkdir(ok_dir)
    print("cp {}/{} {}/{}".format(ARGS.imgdir, name.decode('utf-8'), ok_dir,name.decode('utf-8')))
    copyfile("{}/{}".format(ARGS.imgdir, name.decode('utf-8')), "{}/{}".format(ok_dir,name.decode('utf-8')))

    if False:
        programPause = input("Press the <ENTER> key to continue...")


