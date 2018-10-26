# -*- coding: utf-8 -*-
import os
import re
from nltk import word_tokenize
import codecs

class CharacterBasedFeatures:

	def __init__(self,iC, modelName):
		self.iC = iC
		self.type = "CharacterBasedFeatures"
		self.iC.initFeatureType(self.type)
		self.modelName = modelName
		self.get_uppers()
		self.get_numbers()
		self.get_symbols([","],"commas")
		self.get_symbols(["."],"dots")
		self.get_symbols(['?',"¿"],"questions")
		self.get_symbols(['!','¡'],"exclamations")
		self.get_symbols([":"],"colons")
		self.get_symbols([";"],"semicolons")
		self.get_symbols(['"',"'","”","“", "’"],"quotations")
		self.get_symbols(["—","-","_"],"hyphens")
		self.get_symbols(["(",")"],"parenthesis")
		self.get_in_parenthesis_stats()


	def get_uppers(self):
		for instance in self.iC.instances:
			featValue = 0.0
			matches = re.findall("[A-Z]",instance.text,re.DOTALL)
			upperCases = len(matches)
			ratio = upperCases / len(instance.text)
			instance.addFeature(self.type, self.type+"_UpperCases", ratio)

	def get_in_parenthesis_stats(self):

		for instance in self.iC.instances:
			matches = re.findall("\((.*?)\)", instance.text)
			npar = len(matches)
			totalchars = 0
			totalwords = 0

			for match in matches:
				totalchars += len(match)
				words = word_tokenize(match)
				totalwords = len(words)

			charsInParenthesis = 0.0
			wordsInParenthesis = 0.0
			if npar > 0:
				charsInParenthesis = totalchars / npar
				wordsInParenthesis = totalwords / npar

			instance.addFeature(self.type, self.type+"_charsinparenthesis", charsInParenthesis)
			instance.addFeature(self.type, self.type+"_wordsinparenthesis", wordsInParenthesis)
		

	def get_numbers(self):

		for instance in self.iC.instances:
			matches = re.findall("[0-9]", instance.text)
			ratio = 0.0
			nchars = len(instance.text)

			if nchars > 0:
				ratio = len(matches) / nchars

			instance.addFeature(self.type, self.type+"_Numbers", ratio)
		
		
	def get_symbols(self,symbols, featureName):
		
		for instance in self.iC.instances:
			nChars = len(instance.text)
			matches = 0
			ratio = 0.0
			
			for char in instance.text:
				if char in symbols:
					matches = matches + 1
			
			if nChars > 0:
				ratio = matches / nChars

			instance.addFeature(self.type, self.type+"_"+featureName, ratio)
       