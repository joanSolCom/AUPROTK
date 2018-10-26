from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import scale
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
import csv

import logging

logging.basicConfig(level = logging.INFO, format = '%(levelname)-10s  %(message)s')

class SupervisedLearning:

	def __init__(self, pathTrain, pathTest, pathPredictions, pathGold=None):
		self.trainVectors = []
		self.trainLabels = []

		with open(pathTrain,'r') as tsvin:
			tsvin = csv.reader(tsvin, delimiter='\t')
			for idx, row in enumerate(tsvin):
				if idx > 0:
					vec = list(map(float,row[2].split(",")))
					label = row[3]
					self.trainVectors.append(vec)
					self.trainLabels.append(label)

		self.classifier = self.train()

		strPredictions = "id\ttext\tlabel\n"
		with open(pathTest,'r') as tsvin:
			tsvin = csv.reader(tsvin, delimiter='\t')
			for idx, row in enumerate(tsvin):
				if idx > 0:
					vec = list(map(float,row[2].split(",")))
					predictedLabel = self.classifier.predict([vec])
					strPredictions+=row[0]+"\t"+row[1]+"\t"+predictedLabel[0]+"\n"

		outPred = open(pathPredictions,"w")
		outPred.write(strPredictions)
		outPred.close()

		if pathGold:
			self.evaluate(pathPredictions, pathGold)


	def train(self):
		clf = SVC(kernel="linear")
		clf.fit(self.trainVectors, self.trainLabels)
		return clf


	def evaluate(self, predictions, gold):
		preds = open(predictions,"r").read().strip().split("\n")
		golds = open(gold,"r").read().strip().split("\n")
		i=1
		correct = 0
		nPreds = len(preds)
		while i < nPreds:
			pred = preds[i].split("\t")[2]
			gold = golds[i].split("\t")[2]
			if pred == gold:
				correct+=1
			i+=1

		accuracy = correct / (nPreds - 1)
		logging.info("Accuracy %s",accuracy)
