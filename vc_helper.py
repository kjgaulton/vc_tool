import pandas as pd 
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import sys
import pickle
import sklearn.gaussian_process as gp

from scipy.stats import norm
from scipy.optimize import minimize

	
#error handling classes:
class Error(Exception):
	pass

class FormatError(Error):
	
	def __init__(self, message):
		self.message = message

class vc_helper():

	def __init__(self):
		self.X = ""
		self.Y = ""
		self.pca = ""
		self.seeds = ""
		self.models = ""
		self.gammas = ""
		self.pcs = ""
		self.threads = ""

	#check that the file is formatted correctly
	def verify_bedfile(self, bedfile, training=True):
		print("Verifying format of {}".format(bedfile))
		with open(bedfile) as bed:
			data = bed.readlines()
			header_fields = data[0].split()
			
			#check that the file has a header line:
			if data[0][0] != "#":
				error_message = "Header missing \nfile: {}".format(bedfile)
			
			#check for required columns chrom, start, end, and label
			if training == True:
				if header_fields[:3] + [header_fields[-1]] != ["#chrom", "start", "end", "label"]:
					error_message = "Required header fields missing in training file. #chrom start end should be the first three fields, and label should be the last field. \nfile: {}".format(bedfile)
					raise FormatError(error_message)
			else: 
				if header_fields[:3] != ["#chrom", "start", "end"]:
					error_message = "Required header fields missing in testing file. #chrom start end should be the first three fields, and label should be the last field. \nfile: {}".format(bedfile)
					raise FormatError(error_message)
				
			#get number of fields that we expect from header line
			num_fields = len(data[0].split())
			
			for i, line in enumerate(data[1:]):
				info = line.split()
				
				#verify end >= start
				if (int(info[2]) >= int(info[1])) is False:
					error_message = "Coordinate start position is greater than end position \nfile: {} \nline num: {}".format(bedfile, i+2)
					raise FormatError(error_message)
				
				#check for correct number of features:
				if len(info) != num_fields:
					error_message = "Number of fields inconsistent with header \nfile: {}\nline num: {}".format(bedfile, i+2)
					raise FormatError(error_message)
		
		return data

	def load_dataframe(self, feature_file, training=True):
		from sklearn import preprocessing

		feat_df = pd.read_table(feature_file)

		#reset index
		feat_df["chrom:pos"] = feat_df["#chrom"].map(str) + ":" + feat_df["end"].map(str)
		feat_df.set_index(["chrom:pos"], inplace=True)

		#drop unnecessary columns:
		feat_df.drop(["#chrom", "start", "end"], axis=1, inplace=True)

		#normalize data:
		scaler = preprocessing.StandardScaler()
		if training==True:
			norm = scaler.fit_transform(X=feat_df.drop(["label"], axis=1))
		else: 
			norm = scaler.fit_transform(X=feat_df)

		return feat_df, norm

	def principal_components(self, norm, pca_dataset, n_pcs):
		
		pca = PCA(n_components=n_pcs)
		pca.fit(pca_dataset)
		pca_out = pca.transform(norm)
		return pca_out		

	def sample_loss_grid(self, norm, labels, pca_dataset, gamma, n_pcs, seeds):

		print("evaluating model with gamma={}, num_pcs={}".format(gamma, n_pcs), file=sys.stderr)
		accuracy, precision, aucs = [], [], []
		pca_data = self.principal_components(norm, pca_dataset, n_pcs)
		for seed in seeds:
			a, p, auc, conf = self.evaluate_model(pca_data, labels, gamma, seed)
			accuracy.append(a)
			precision.append(p)
			aucs.append(auc)

		model = LabelPropagation(
			kernel='rbf',
			gamma=gamma,
			max_iter=1000000
		).fit(pca_data, labels)

		return [np.mean(aucs), np.mean(precision), np.mean(accuracy), model]

	def sample_loss_bayesian(self, p):
		accuracy, precision, aucs = [], [], []

		gamma = p[0]
		num_pcs = int(p[1])
		print("evaluating model with gamma={}, num_pcs={}".format(gamma, num_pcs), file=sys.stderr)

		pca_data = self.principal_components(self.X, self.pca, num_pcs)
		for seed in self.seeds:
			a, p, auc, conf = self.evaluate_model(pca_data, self.Y, gamma, seed)
			accuracy.append(a)
			precision.append(p)
			aucs.append(auc)

		model = LabelPropagation(
			kernel='rbf',
			gamma=gamma,
			max_iter=1000000
		).fit(pca_data, self.Y)

		#return score
		return [np.mean(aucs), np.mean(precision), np.mean(accuracy), model]

	def evaluate_model(self, X, Y, gamma, seed, max_iter=100000):
		#set random seed:
		np.random.seed(seed)

		X_train, X_test, Y_train, Y_test = train_test_split(
			X,
			Y,
			stratify = Y,
			test_size = 0.20,
			random_state = seed
		)

		lp_model = LabelPropagation(
			kernel = 'rbf', 
			gamma=gamma,
			max_iter = max_iter
		)

		lp_model.fit(X_train, Y_train)

		#test model on validation data
		predicted_labels = lp_model.predict(X_test)
		predicted_prob = lp_model.predict_proba(X_test)

		#get just the labeled testing data:
		labeled_prob = [p[1] for i, p in enumerate(predicted_prob) if Y_test[i] in [0, 1]]
		labels = [p for i, p in enumerate(predicted_labels) if Y_test[i] in [0, 1]]
		true_labels = [l for l in Y_test if l in [0,1]]

		#evaluation
		accuracy = metrics.accuracy_score(true_labels, labels)
		precision = metrics.precision_score(true_labels, labels)
		auc = metrics.roc_auc_score(true_labels, labeled_prob)
		conf = metrics.confusion_matrix(true_labels, labels)

		return accuracy, precision, auc, conf

	def grid_search(self, parameters):
		print("start running grid search", file=sys.stderr)

		with Pool(self.threads) as pool:
			loss = pool.starmap(self.sample_loss_grid, parameters)

		print("finished running grid search", file=sys.stderr)
		return loss

	def bayesian_search(self, parameters):
		print("start running bayesian search", file=sys.stderr)
		bounds = np.array([self.gammas, self.pcs])
		seeds = self.seeds

		xp, yp = self.bayesian_optimisation(
			n_iters=8,
			sample_loss=self.sample_loss_bayesian,
			bounds = bounds,
			n_pre_samples=1,
			random_search=100000
		)
		print("finished running bayesian search", file=sys.stderr)
		return xp, yp

	def testing_predictions(self, test_data, model, num_pcs, gamma=False, max_iter=1000000, mean=False):

		pca_data = self.principal_components(test_data, self.pca, num_pcs)
		train_pca_data = self.principal_components(self.X, self.pca, num_pcs)

		if mean == False:
			return np.array([p[1] for p in model.predict_proba(pca_data)])

		predicted_probs = ""
		for seed in self.seeds:
			np.random.seed(seed)

			model = LabelPropagation(
				kernel = 'rbf', 
				gamma=gamma,
				max_iter = max_iter
			)
			model.fit(train_pca_data, self.Y)

			predicted_prob = np.array([p[1] for p in model.predict_proba(pca_data)])
			if predicted_probs == "":
				predicted_probs = predicted_prob
			else:
				predicted_probs = np.vstack((predicted_probs, predicted_prob))

		#get mean of each run:
		mean_probs = np.mean(predicted_probs, axis=0)
		return mean_probs


	def expected_improvement(self, x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):

		x_to_predict = x.reshape(-1, n_params)

		mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

		if greater_is_better:
			loss_optimum = np.max(evaluated_loss)
		else:
			loss_optimum = np.min(evaluated_loss)

		scaling_factor = (-1) ** (not greater_is_better)

		# In case sigma equals zero
		with np.errstate(divide='ignore'):
			Z = scaling_factor * (mu - loss_optimum) / sigma
			expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
			expected_improvement[sigma == 0.0] == 0.0

		return -1 * expected_improvement


	def sample_next_hyperparameter(self, acquisition_func, gaussian_process, evaluated_loss, greater_is_better=False,
								   bounds=(0, 10), n_restarts=25):

		best_x = None
		best_acquisition_value = 1
		n_params = bounds.shape[0]

		for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):

			res = minimize(fun=acquisition_func,
						   x0=starting_point.reshape(1, -1),
						   bounds=bounds,
						   method='L-BFGS-B',
						   args=(gaussian_process, evaluated_loss, greater_is_better, n_params))

			if res.fun < best_acquisition_value:
				best_acquisition_value = res.fun
				best_x = res.x

		return best_x


	def bayesian_optimisation(self, n_iters, sample_loss, bounds, x0=None, n_pre_samples=5,
							  gp_params=None, random_search=False, alpha=1e-5, epsilon=1e-7):

		x_list = []
		y_list = []

		n_params = bounds.shape[0]

		if x0 is None:
			#for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pre_samples, bounds.shape[0])):
			for params in np.hstack((np.random.uniform(bounds[0][0], bounds[0][1], (n_pre_samples, 1)), np.random.randint(bounds[1][0], bounds[1][1], (n_pre_samples, 1)))):
				x_list.append(params)
				y_list.append(sample_loss(params))
		else:
			for params in x0:
				x_list.append(params)
				y_list.append(sample_loss(params))

		xp = np.array(x_list)
		yp = np.array(y_list)

		# Create the GP
		if gp_params is not None:
			model = gp.GaussianProcessRegressor(**gp_params)
		else:
			kernel = gp.kernels.Matern()
			model = gp.GaussianProcessRegressor(kernel=kernel,
												alpha=alpha,
												n_restarts_optimizer=10,
												normalize_y=True)

		for n in range(n_iters):

			model.fit(xp, yp[:,0])
			# Sample next hyperparameter
			if random_search:
				x_random = np.hstack((np.random.uniform(bounds[0][0], bounds[0][1], size=(random_search, 1)), np.random.randint(bounds[1][0], bounds[1][1], size=(random_search, 1)))) 
				ei = -1 * self.expected_improvement(x_random, model, yp[:,0], greater_is_better=True, n_params=n_params)
				next_sample = x_random[np.argmax(ei), :]
			else:
				next_sample = sample_next_hyperparameter(self.expected_improvement, model, yp[:,0], greater_is_better=True, bounds=bounds, n_restarts=100)

			# Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
			if np.any(np.abs(next_sample - xp) <= epsilon):
				next_sample = [np.random.uniform(bounds[0][0], bounds[0][1]), np.random.randint(bounds[1][0], bounds[1][1])]
			# Sample loss for new set of parameters
			cv_score = sample_loss(next_sample)
			# Update lists
			x_list.append(next_sample)
			y_list.append(cv_score)

			# Update xp and yp
			xp = np.array(x_list)
			yp = np.array(y_list)

		return xp, yp







