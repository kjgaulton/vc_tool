#!/usr/bin/env python3

import numpy as np 
import pandas as pd 
import sys
import os
import time
import argparse
import errno
from vc_helper import vc_helper
import pickle

vch = vc_helper()

#error handling classes:
class Error(Exception):
	pass

class OutputPathError(Error):
	def __init__(self, message):
		self.message = message

class OptionError(Error):
	def __init__(self, message):
		self.message = message


#initialize argument parser:
parser = argparse.ArgumentParser()
required = parser.add_argument_group('required arguments')
required.add_argument('-t', '--train', help='path to training feature file, or trained model (.model extension)', required=True)
required.add_argument('-o', '--outdir', help='path to directory where output will be stored', required=True)
required.add_argument('-n', '--name', help='prefix of output file names', required=True)
required.add_argument('-p', '--pcadata', help='path to pca data feature matrix', required=True)
parser.add_argument('-x', '--optimize', help='method of hyperparameter optimization, grid or bayesian')
parser.add_argument('-a', '--pcavalues', help='minimum and maximum number of PCs to test(step size = 1). Example: -a 10 15. \n Use one value if model is already trained.', nargs='+', type=int)
parser.add_argument('-g', '--gammas', help='minimum, maximum, and step size for RBF kernel gamma parameter. Example: -g 0.1 5 0.1. \n Use one value if model is already trained.', nargs='+', type=float)
parser.add_argument('-s', '--test', help='path to testing feature file')
parser.add_argument('-m', '--mean', help='input an integer n. predictions are comprised of the mean of n models with n random seeds. \n Must also include training file when using this flag.', type=int)
parser.add_argument('-d', '--numseeds', help='integer number of random seeds to evaluate during training. default = 25', type=int, default=25)
parser.add_argument('-c', '--threads', help='number of threads for multithreading, default = 8', type=int, default=8 )
parser.add_argument('-b', '--biters', help='number of iterations for bayesian optimization, default=25', type=int, default=25)
args = parser.parse_args()

#intro message:
print("###############################################################################################")
print("# Semi-supervised Predictive frAmework for Genetic variants using Epigenome feaTures (SPAGET) #")
print("# Author of this software: Samuel Morabito                                                    #")
print("# Gaulton Lab 2018 http://www.gaultonlab.org/                                                 #")
print("# UC San Diego                                                                                #")
print("# https://github.com/smorabit/vc_tool                                                         #")
print("###############################################################################################")
print()

#print to stderr all of the arguments that are used:


train_file = args.train
test_file = args.test 
pca_file = args.pcadata
outdir = args.outdir 
name = args.name
optimize = args.optimize
num_seeds = args.numseeds

#are we using a model that has already been trained?
if train_file.endswith(".model"):
	trained = True
else:
	trained = False

#check hyperparameter optimization option
if optimize not in ["grid", "bayesian", None]:
	raise OptionError("invalid option for -x or --optimize flag, select either grid or bayesian")

#can we write to output destination
if os.path.exists(outdir + name):
	pass
elif os.access(os.path.dirname(outdir + name), os.W_OK):
	pass
else:
	print("cannot write file here", file=sys.stderr)
	raise OutputPathError("permission not granted to  write output files here")

#verify the file format of the training and testing files:
if trained == False:
	vch.verify_bedfile(train_file, training=True)
if test_file != None:
	vch.verify_bedfile(test_file, training=False)
if pca_file != None:
	vch.verify_bedfile(pca_file, training=False)

#load data into dataframes
if trained == False:
	train_df, train_norm = vch.load_dataframe(train_file)
if test_file != None:
	test_df, test_norm = vch.load_dataframe(test_file, training=False)
pca_df, pca_norm = vch.load_dataframe(pca_file, training=False)
vch.pca = pca_norm

if trained == False:
	vch.X = train_norm
	vch.Y = train_df['label']
if args.gammas != None:
	vch.gammas = np.arange(*args.gammas)
vch.biters = args.biters
vch.seeds = [np.random.randint(1,10000000) for i in range(num_seeds)]
vch.pcs = [i for i in range(*args.pcavalues)]
vch.threads = args.threads

#set trained to true if we are looking at the mean and only one param value is chosen
if len(args.pcavalues) == 1 and args.mean != None:
	trained = True

#grid search optimization
if optimize == "grid" and trained == False:
	parameters = [[train_norm, train_df['label'], pca_norm, gamma, n_pcs, vch.seeds] for gamma in vch.gammas for n_pcs in vch.pcs]
	loss = vch.grid_search(parameters)
	print("done running grid search", file=sys.stderr)
	print("writing performance to {}".format(outdir+name+".performance"), file=sys.stderr)
	with open(outdir + name + ".performance", 'w') as f:
		f.write("\t".join(["gamma", "num_pcs", "roc_auc", "precision", "accuracy"]) + "\n")
		for i, p in enumerate(parameters):
			f.write("\t".join([str(p[3]), str(p[4]), str(loss[i][0]), str(loss[i][1]), str(loss[i][2])]) + "\n")
	best = [l[0] for l in loss].index(max([l[0] for l in loss]))
	best_model = loss[best][-1]
	best_params = parameters[best][3:-1]
	print("best:", loss[best][0])
	print("params:", parameters[best][3:-1])
	print("pickle best model to file", outdir+name+".model")
	pickle.dump(best_model, open(outdir+name+".model", 'wb'))

#bayesian optimization with gaussian process
elif optimize == "bayesian" and trained == False:
	params, scores = vch.bayesian_search("")
	print("params:", params)
	print("scores:", scores)
	aucs = scores[:,0]
	precision = scores[:,1]
	accuracy = scores[:,2]
	models = scores[:,3]
	print("writing performance to {}".format(outdir+name+".performance"), file=sys.stderr)
	with open(outdir+name+".performance", 'w') as f:
		f.write("\t".join(["gamma", "num_pcs", "roc_auc", "precision", "accuracy"]) + "\n")
		for i, p in enumerate(params):
			f.write("\t".join([str(p[0]), str(p[1]), str(aucs[i]), str(precision[i]), str(accuracy[i])]) + "\n")
			
	maxind = np.argmax(scores[:,0])
	best_model = models[maxind]
	best_params = params[maxind]
	print("best", scores[:,0][maxind])
	print("params:", params[maxind])
	print("pickling best model:")
	pickle.dump(best_model, open(outdir+name+".model", 'wb'))

if test_file != None:

	#are we using a model that we just made or are we loading a model?
	if train_file.endswith(".model"):
		best_model = pickle.load(open(train_file, 'rb'))
		test_pcs = args.pcavalues[0]
	elif args.mean != None:
		test_pcs = args.pcavalues[0]
		best_model = ""
	else:
		test_pcs = int(best_params[1])

	#make predictions on testing set:
	print("make predictions on testing set:", file=sys.stderr)
	if args.mean == None:
		test_predictions = vch.testing_predictions(test_norm, best_model, test_pcs)
	else:
		test_predictions = vch.testing_predictions(test_norm, best_model, test_pcs, args.gammas[0], mean=True)

	#write predictions to file:
	print("writing predictions to file", file=sys.stderr)
	coords = list(test_df.index)

	print("length of coords:", len(coords))
	print("length of predictions:", len(test_predictions))
	with open(outdir+name+"_predictions.bed", 'w') as f:
		for i, coord in enumerate(coords):
			c = coord.split(":")
			out = "\t".join([c[0], c[1], c[1], str(test_predictions[i])]) + "\n"
			f.write(out)


