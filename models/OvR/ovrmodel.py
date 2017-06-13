import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

xtrain = pd.read_pickle(open('X_train.pickle', 'rb'))
ytrain = pd.read_pickle(open('y_train.pickle', 'rb'))

xtrain = xtrain.iloc[:,-100:]
xtrain = xtrain.values

#model.fit(xtrain, ytrain)

def main():
	model = OneVsRestClassifier(LinearSVC(random_state=1), n_jobs=-1)

	filename = 'OvRmodel.sav'
	pickle.dump(model, open(filename, 'wb'))

if __name__ == "__main__":
	main()