# -*- coding: utf-8 -*-
import instanceManager
import logging
from instanceManager import InstanceCollection
from featureClasses.characterBasedFeatures import CharacterBasedFeatures
from featureClasses.wordBasedFeatures import WordBasedFeatures
from featureClasses.sentenceBasedFeatures import SentenceBasedFeatures
from featureClasses.dictionaryBasedFeatures import DictionaryBasedFeatures
from featureClasses.syntacticFeatures import SyntacticFeatures
from pprint import pprint
import sys
import os

logging.basicConfig(level = logging.INFO, format = '%(levelname)-10s  %(message)s')

def compute_features(pathInput, modelName="default"):
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

if __name__ == "__main__":
	if len(sys.argv) != 3:
		logging.error("Wrong number of parameters.")
		logging.error("This is the correct way: python3 main.py path/to/inputfile path/to/outputFileWhichIsCreatedByTheProgram")
	else:
		inputFile = sys.argv[1]
		if os.path.isfile(inputFile) and not os.path.isdir(inputFile):
			outputFile = sys.argv[2]
			logging.info("Computing features")
			iC = compute_features(inputFile)
			jsonStr = iC.generateJSONOutput()
			logging.info("Writing Features in %s",outputFile)
			with open(outputFile, "w") as fd:
				fd.write(jsonStr)
				fd.close()
			
			logging.info("Features Correctly Written")
		
		else:
			logging.error("Incorrect input file. Please check that %s, is an existing file and not a directory.", inputFile)
			logging.error("./input/sampleInput.txt is a correct input file. Use it as inspiration.")