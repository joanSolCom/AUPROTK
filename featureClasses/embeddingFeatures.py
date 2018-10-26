from gensim import models
import numpy as np
import os
from collections import defaultdict

class EmbeddingFeatures:

	def __init__(self,iC, modelName, modelPath):
		self.iC = iC
		self.type = "EmbeddingFeatures"
		self.iC.initFeatureType(self.type)
		self.modelName = modelName
		self.modelPath = modelPath

	def cleanup(self):
		self.model = None
		self.w2v = None
		self.dim = None

	def getWordVector(self, word):
		vector = np.zeros(self.dim)

		if word in self.w2v:
			vector = self.w2v[word]

		return vector

	def getW2V(self):
		return self.w2v

	def computeSimilarity(self, word1, word2):
		return self.model.similarity(word1, word2)

	def getAvgVectors(self):
		self.model = models.KeyedVectors.load_word2vec_format(self.modelPath, binary=True)
		self.w2v = dict(zip(self.model.wv.index2word, self.model.wv.syn0))
		self.dim = len(self.w2v.itervalues().next())

		vectors = []

		for instance in self.iC.instances:
			words = instance.tokens

			for word in words:
				vector = self.getWordVector(word)
				vectors.append(vector)

			avgVector = np.mean(vectors,axis=0)
			for idxWord, dim in enumerate(avgVector):
				instance.addFeature(self.type, self.type+"_embed"+str(idxWord), dim)

