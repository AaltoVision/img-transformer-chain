import csv
import argparse
import numpy as np

PARSER = argparse.ArgumentParser()
PARSER.add_argument("-f", action='store', required=True, dest='fname')
PARSER.add_argument("-srcdir", action='store', required=True, dest='srcdir')
ARGS = PARSER.parse_args()

def get_ages(args):
    """ Eval the ages"""
    i = 1

    all_ages = {}

    with open(ARGS.fname, "rt", encoding='utf-8') as f:
        freader = csv.reader(f, delimiter=",")
        for row in freader:
            try:
                if len(row) > 1:
                    all_ages[row[0].encode('utf-8')] = row[1].encode('utf-8')
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
	print("Mean age of {0} samples is {1:0.2f} y with std = {2:0.3f}".format(len(age_vals), np.mean(age_vals), np.std(age_vals)))
	f = open('mean_ages.txt','a')
	f.write('{0},{1:0.2f},{2:0.3f}\n'.format(ARGS.srcdir, np.mean(age_vals), np.std(age_vals)))
	f.close()

get_mean(get_ages(ARGS))
