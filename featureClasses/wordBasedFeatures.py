# -*- coding: utf-8 -*-
import os
import nltk
import re
from math import fabs
from pprint import pprint
import codecs
import numpy as np

class WordBasedFeatures:
			
	def __init__(self,iC, modelName):
		self.iC = iC
		self.type = "WordBasedFeatures"
		self.iC.initFeatureType(self.type)
		self.modelName = modelName
		self.get_twothree_words()
		self.get_word_stdandrange()
		self.get_chars_per_word()
		self.get_vocabulary_richness()
		self.get_stopwords()
		self.get_acronyms()
		self.get_firstperson_pronouns()
		self.get_proper_nouns()
		self.getNERatio()
		
	def get_twothree_words(self):

		for instance in self.iC.instances:
			textTokenized = instance.tokens
			nwords = len(textTokenized)
			nTwo = 0
			nThree = 0
			twoWords = 0.0
			threeWords = 0.0

			for word in textTokenized:
				if len(word) == 2:
					nTwo +=1
				elif len(word) == 3:
					nThree+=1

			if nwords > 0:
				twoWords = nTwo / nwords
				threeWords = nThree / nwords

			instance.addFeature(self.type, self.type+"_twoWords", twoWords)
			instance.addFeature(self.type, self.type+"_threeWords", threeWords)

	def get_word_stdandrange(self):

		for instance in self.iC.instances:
			textTokenized = instance.tokens
			lengths = []
			for word in textTokenized:
				lengths.append(len(word))
			std = np.std(lengths)
			rng = np.amax(lengths) - np.amin(lengths)
			
			instance.addFeature(self.type, self.type+"_STD", std)
			instance.addFeature(self.type, self.type+"_Range", rng)

	def get_chars_per_word(self):
		
		for instance in self.iC.instances:
			lWords = instance.tokens
			nwords = len(lWords)
			ratio = 0.0
			
			ncharsword = 0

			for word in lWords:		
				nchars = len(word)
				ncharsword = ncharsword + nchars
			
			if nwords > 0:
				ratio = ncharsword / nwords

			instance.addFeature(self.type, self.type+"_CharsPerWord", ratio)

	def get_vocabulary_richness(self):
		
		for instance in self.iC.instances:
			lAllWords = instance.lemmas
			lDiffWords = set(lAllWords)
			ratio = 0.0
			if len(lAllWords) > 0:
				ratio = len(lDiffWords) / len(lAllWords)
			instance.addFeature(self.type, self.type+"_VocabularyRichness", ratio)

	def get_stopwords(self):

		stopwords = nltk.corpus.stopwords.words('english')
		for instance in self.iC.instances:					
			words = instance.tokens
			nstopwords = 0
			totalWords = 0
			ratio = 0.0
			for word in words:
				totalWords = totalWords + 1
				if word.strip().lower() in stopwords:
					nstopwords = nstopwords + 1		
			
			if len(words) > 0:
				ratio = nstopwords / totalWords

			instance.addFeature(self.type, self.type+"_Stopwords", ratio)

	def get_acronyms(self):

		for instance in self.iC.instances:
			nacr = 0
			words = instance.tokens
			nwords = len(words)
			totalWords = 0
			ratio = 0.0
			for word in words:
				totalWords = totalWords + 1	
				pattern = '(^[A-Z]([0-9]|[A-Z]|\.){3})'
				match = re.match(pattern, word)
				if match and word[len(word) - 1] != ":" and word[len(word) -1] != ',':
					nacr = nacr + 1
			
			if nwords > 0:
				ratio = nacr / totalWords

			instance.addFeature(self.type, self.type+"_Acronyms", ratio)

	def get_firstperson_pronouns(self):

		first_singular = ["i","me","my","mine"]
		first_plural = ["we","our","ours"]

		for instance in self.iC.instances:
			lWords = instance.tokens
			nwords = len(lWords)
			ratioFirstS = 0.0
			ratioFirstP = 0.0

			nFirstS = 0
			nFirstP = 0
			for word in lWords:
				word = word.lower()
				if word in first_singular:
					nFirstS = nFirstS + 1
				elif word in first_plural:
					nFirstP = nFirstP + 1
			
			if nwords > 0:
				ratioFirstS = nFirstS / nwords
				ratioFirstP = nFirstP / nwords

			instance.addFeature(self.type, self.type+"_FirstSingular", ratioFirstS)
			instance.addFeature(self.type, self.type+"_FirstPlural", ratioFirstP)
	
	def get_proper_nouns(self):

		for instance in self.iC.instances:
			nwords = len(instance.tokens)
			sentences = instance.sentenceTokens
			proper = 0
			ratio = 0.0
			for sentence in sentences:
				first = False
				for word in sentence:
					if first == False:
						first = True
					else:
						if word[0].isupper():
							proper = proper + 1	
			if nwords > 0:				
				ratio = proper / nwords

			instance.addFeature(self.type, self.type+"_ProperNouns", ratio)


	def getNERatio(self):
		for instance in self.iC.instances:
			nwords = len(instance.tokens)
			nNe = len(instance.namedEntities)
			ratio = 0
			if nwords > 0:				
				ratio = nNe / nwords

			instance.addFeature(self.type, self.type+"_NERatio", ratio)

	def getUrlRatio(self):
		for instance in self.iC.instances:
			nwords = len(instance.tokens)
			nNe = len(instance.namedEntities)
			ratio = 0
			if nwords > 0:				
				ratio = nNe / nwords

			instance.addFeature(self.type, self.type+"_NERatio", ratio)