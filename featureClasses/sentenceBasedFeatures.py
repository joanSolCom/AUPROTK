# -*- coding: utf-8 -*-
from __future__ import division
import codecs
from nltk import word_tokenize
import numpy as np
import os

class SentenceBasedFeatures:

	def __init__(self,iC, modelName):
		self.iC = iC
		self.type = "SentenceBasedFeatures"
		self.iC.initFeatureType(self.type)
		self.modelName = modelName
		self.get_wordsPerSentence_stdandrange()
			
	def get_wordsPerSentence_stdandrange(self):
		for instance in self.iC.instances:
			sentences = instance.sentences
			lengths = []
			for sentence in sentences:
				lengths.append(len(word_tokenize(sentence)))
			
			std = np.std(lengths)
			mean = np.mean(lengths)
			rng = np.amax(lengths) - np.amin(lengths)

			instance.addFeature(self.type, self.type+"_STD", std)
			instance.addFeature(self.type, self.type+"_Range", rng)
			instance.addFeature(self.type, self.type+"_wordsPerSentence", mean)
