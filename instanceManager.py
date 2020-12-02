from featureManager import FeatureSet
import os
import numpy as np
from nltk import word_tokenize
import codecs
import nltk
import spacy
import json
import logging
from numpyencoder import NumpyEncoder

logging.basicConfig(level = logging.INFO, format = '%(levelname)-10s  %(message)s')

logging.info("Loading Spacy...")
en_nlp = spacy.load('en_core_web_lg')
logging.info("Spacy correctly loaded")

def createInstanceCollection(pathInput):
	iC = InstanceCollection()

	with open(pathInput) as file_with_json_lines:
		for row in file_with_json_lines:
			if row.strip():
				data = json.loads(row)
				instance = Instance(data["tweet_id"], None, data["text"])
				iC.addInstance(instance)

	return iC


class Instance:

	def __init__(self, name, label, text):
		self.name = name
		self.featureSet = FeatureSet()
		self.label = label
		
		self.text = text.replace("\"","'").replace("\\","")
		self.text = " ".join(self.text.split())
		self.namedEntities = []
		self.tokens = []
		self.lowerTokens = []
		self.lemmas = []
		self.pos = []
		self.urls = []
		self.posSimple = []
		self.sentences = []
		self.sentenceTokens = []
		self.morph = []
		self.process()

		'''
		i=0
		while i<len(self.morph):
			print(self.tokens[i])
			print(self.morph[i])
			i+=1
		'''
		
	def process(self):
		doc = en_nlp(self.text)
		for ent in doc.ents:
			self.namedEntities.append((ent.text,ent.label_))

		textConll = ""
		for sent in doc.sents:
			s = []
			self.sentences.append(sent.text)
			sentConll = ""
			for token in sent:
				s.append(token.text)
				lineConll = str(token.i - sent.start + 1)+"\t"+token.text+"\t"+token.lemma_+"\t"+token.tag_+"\t"+token.dep_+"\t"+str(token.head.i - sent.start + 1)+"\n"
				sentConll+=lineConll

			self.sentenceTokens.append(s)
			textConll+=sentConll+"\n"

		self.conll = textConll

		for token in doc:
			self.posSimple.append(token.pos_)
			self.pos.append(token.tag_)
			self.tokens.append(token.text)
			self.lowerTokens.append(token.text.lower())
			self.lemmas.append(token.lemma_)
			if token.like_url:
				self.urls.append(token.text)

			morphDict = en_nlp.vocab.morphology.tag_map[token.tag_]
			self.morph.append(morphDict)

	def getFeatures(self):
		return self.featureSet.getFeatures()

	def getFeaturenames(self, featuresSelected):
		return self.featureSet.getFeaturenames(featuresSelected)

	def getFeatureTypeNames(self, featuresSelected=None):
		return self.featureSet.getFeatureTypeNames(featuresSelected)

	def getFeatureVector(self, featuresSelected):
		return self.featureSet.getFeatureVector(featuresSelected)

	def initFeatureType(self, featureType):
		self.featureSet.initFeatureType(featureType)

	def addFeature(self, featureType, featureName, featureValue):
		self.featureSet.addFeature(featureType, featureName, featureValue)

	def updateFeature(self, featureType, featureName, increment, operation="sum"):
		self.featureSet.updateFeature(featureType, featureName, increment, operation)

	def __repr__(self):
		return self.name + "\n" + self.label+ "\n" + str(self.featureSet) #+ "\n"+str(self.tokens)

class InstanceCollection:

	def __init__(self):
		self.instances = []
		self.labels = set()
		self.instanceDict = {}

	def __repr__(self):
		strCollection = ""
		for instance in self.instances:
			strCollection += "---------\n"+ str(instance) +"\n---------"
		return strCollection

	def initFeatureType(self, featureType):
		for instance in self.instances:
			instance.initFeatureType(featureType)

	def addInstance(self, instance):
		self.instances.append(instance)
		self.instanceDict[instance.name] = instance
		self.labels.add(instance.label)


	def getFeatureNames(self, featuresSelected):
		return self.instances[0].getFeatureNames(featuresSelected)

	def getFeatureTypeNames(self, featuresSelected=None):
		return self.instances[0].getFeatureTypeNames(featuresSelected)
	
	def generateJSONOutput(self):
		jsonList = []
		for instance in self.instances:
			jsonDict = {}
			featureDict = instance.getFeatures()
			jsonDict["features"] = featureDict
			jsonDict["tweet_id"] = instance.name
			jsonList.append(jsonDict)
		
		return json.dumps(jsonList, cls=NumpyEncoder)

	def generateTSV(self, path=None):
		featureTypeNames = self.getFeatureTypeNames()

		strOut = "id\ttext\tfeatures\tlabel\n"
		for instance in self.instances:
			featureVector = instance.getFeatureVector(featureTypeNames)
			strVec = ",".join(map(str, featureVector))
			text = instance.text
			strOut+=instance.name+"\t"+instance.text+"\t"+strVec+"\t"+instance.label+"\n"

		if path:
			out = open(path,"w")
			out.write(strOut)
			out.close()

		return strOut


	def getSklearnInput(self, featuresSelected = None):
		X = []
		Y = []

		featureTypeNames = self.getFeatureTypeNames(featuresSelected)

		for instance in self.instances:
			featureVector = instance.getFeatureVector(featureTypeNames)
			X.append(featureVector)
			Y.append(instance.label)

		return X, Y

	def getMeanFeatValuesPerClass(self, featuresSelected=None):
		featureTypeNames = self.getFeatureTypeNames(featuresSelected)
		nFeats = len(featureTypeNames)

		dictPerClass = {}

		for instance in self.instances:
			featureVector = instance.getFeatureVector(featureTypeNames)
			label = instance.label
			if label not in dictPerClass:
				dictPerClass[label] = np.array([featureVector],dtype=np.float64)
			else:
				dictPerClass[label] = np.append(dictPerClass[label],[featureVector],axis=0)

		outDict = {}
		for label, matrix in dictPerClass.items():
			i=0
			outDict[label] = {}
			while i < nFeats:
				
				featureValues = matrix[:,i]
				featureType, featureName = featureTypeNames[i]
				
				mean = np.mean(featureValues)
				median = np.median(featureValues)
				std = np.std(featureValues)
				
				outDict[label][featureName] = {}
				outDict[label][featureName]["mean"] = mean
				outDict[label][featureName]["median"] = median
				outDict[label][featureName]["std"] = std

				i+=1

		return outDict


	
		
	