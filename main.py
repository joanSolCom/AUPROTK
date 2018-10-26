# -*- coding: utf-8 -*-
import instanceManager
import logging
from instanceManager import InstanceCollection
from featureClasses.characterBasedFeatures import CharacterBasedFeatures
from featureClasses.wordBasedFeatures import WordBasedFeatures
from featureClasses.sentenceBasedFeatures import SentenceBasedFeatures
from featureClasses.dictionaryBasedFeatures import DictionaryBasedFeatures
from featureClasses.syntacticFeatures import SyntacticFeatures
from featureClasses.lexicalFeatures import LexicalFeatures
from pprint import pprint
from machineLearning.classify import SupervisedLearning

logging.basicConfig(level = logging.INFO, format = '%(levelname)-10s  %(message)s')

def compute_features(pathInput, modelName):
	logging.info("Creating Instance Collection")
	iC = instanceManager.createInstanceCollection(pathInput)
	
	logging.info("Character based")
	iChar = CharacterBasedFeatures(iC,modelName)
	
	logging.info("Word based")
	iWord = WordBasedFeatures(iC,modelName)
	
	logging.info("Sentence based")
	iSent = SentenceBasedFeatures(iC,modelName)

	logging.info("Dictionary based")
	iDict = DictionaryBasedFeatures(iC,modelName)
	
	logging.info("Syntactic features")
	iSyntactic = SyntacticFeatures(iC,modelName)
	
	return iC


###Generate training data
modelName = "sample"
inPathTrain = "./input/train.tsv"
featurePathTrain = "./features/train.tsv"

logging.info("Computing features training set")
iC = compute_features(inPathTrain, modelName)
logging.info("Generating Output")
iC.generateTSV(featurePathTrain)

inPathTest = "./input/test.tsv"
featurePathTest = "./features/test.tsv"

logging.info("Computing features test set")
iC = compute_features(inPathTest, modelName)
logging.info("Generating Output")
iC.generateTSV(featurePathTest)

predictionPath = "./predictions/results.tsv"
logging.info("Classifying")
iSup = SupervisedLearning(featurePathTrain, featurePathTest, predictionPath, inPathTest)

