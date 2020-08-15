import h2o4gpu as sklearn
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import pickle
from loadData import formatData

PATH = "processedData2a/"
num_subjects = 9

try:
	with open("dataset_2a.pickle", "rb") as f:
		X_train, Y_train, X_test, Y_test = pickle.load(f)

except:
	# initializing training and testing arrays with the first subject
	X_train, Y_train = formatData(PATH, 1, True)
	X_test, Y_test = formatData(PATH, 1, False)

	for i in range(2, num_subjects+1):
		x_train, y_train = formatData(PATH, i, True)
		x_test, y_test = formatData(PATH, i, False)

		X_train = np.append(X_train, x_train, axis=0)
		Y_train = np.append(Y_train, y_train, axis=0)
		X_test = np.append(X_test, x_test, axis=0)
		Y_test = np.append(Y_test, y_test, axis=0)

	print(X_train.shape, Y_train.shape)
	print(X_test.shape, Y_test.shape)

	with open("dataset_2a.pickle", "wb") as f:
		pickle.dump((X_train, Y_train, X_test, Y_test), f)


# Flatten the last dimension
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# print(X_train.shape)
# print(X_test.shape)

try:
	with open("currentModel.pickle", "rb") as f:
		best_clf = pickle.load(f)

except:
	model = sklearn.svm.SVC(kernel="linear")

	# ------------------------ GRIDSEARCHCV ------------------
	C_values = np.logspace(-2, 3, 20)
	gamma_values = np.logspace(-3, 1, 20)

	parameters = [{'C': C_values, 'kernel': ['linear']},
				{'C': C_values, 'kernel': ['rbf'], 'gamma': gamma_values}]

	grid_search = GridSearchCV(estimator=model, param_grid=parameters, scoring='accuracy', n_jobs=-1)
	grid_search = grid_search.fit(X_train, Y_train)

	print(grid_search.best_score_)
	print(grid_search.best_params_)

	# obtain the best model 
	best_clf = grid_search.best_estimator_

	with open("currentModel.pickle", "wb") as f:
		pickle.dump((best_clf), f)


Y_pred = best_clf.predict(X_test)
acc = sklearn.metrics.accuracy_score(Y_test, Y_pred)
print("Prediction accuracy of the testing dataset: {}".format(acc))