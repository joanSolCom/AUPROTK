import codecs
import numpy as np
from TreeLib.treeOperations import SyntacticTreeOperations
import os

class SyntacticFeatures:

	adverbialRelations = ["advcl","advmod","npmod"]
	modifierRelations = ["nounmod","neg","meta","poss","prep","quantmod","relcl","nummod","npmod"]
	passiveRelations = ["auxpass", "csubjpass","nsubjpass"]

	verbTags = ["VB","VBD","VBG","VBN","VBP","VBZ", "MD"]
	nounTags = ["NN","NNS","NNP","NNPS"]
	adverbTags = ["RB","RBR","RBS","WRB"]
	adjectiveTags = ["JJ","JJR","JJS"]
	pronounTags = ["PRP","PRP$","WP","WP$"]
	determinerTags = ["DT","PDT","WDT"]
	conjunctionTags = ["CC","IN"]

	superlatives = ["JJS","RBS"]
	comparatives = ["JJR","RBR"]
	
	pastVerbs = ["VBD","VBN"]
	presentVerbs = ["VBG","VBP","VBZ"]
	

	def __init__(self,iC, modelName):	
		
		self.iC = iC
		self.type = "SyntacticFeatures"
		self.iC.initFeatureType(self.type)
		self.allRelationsPos = open("./dicts/allRelationsPos.txt","r").read().split("\n")
		self.modelName = modelName
		self.compute_syntactic_features()

	def compute_syntactic_features(self):

		nPosts = len(self.iC.instances)
		nProcessed = 0

		for instance in self.iC.instances:
			conllSents = instance.conll.split("\n\n")
			iTrees = []
			conllSents = conllSents[:-1]
			for conllSent in conllSents:
				try:
					iTree = SyntacticTreeOperations(conllSent)
					iTrees.append(iTree)
				except ValueError as e:
					print(e)
					continue

			self.get_relation_usage(iTrees, instance)
			self.get_relationgroup_usage(iTrees, instance)
			self.get_pos_usage(iTrees, instance)
			self.get_posgroup_usage(iTrees, instance)
			self.getPassiveUsage(iTrees,instance)
			self.get_shape_features(iTrees, instance)
			self.get_verb_features(iTrees, instance)
			nProcessed +=1

		self.adjust_features()

	#to be used after get_relation_usage and get_pos_usage
	def adjust_features(self):
		for instance in self.iC.instances:
			for featName in self.allRelationsPos:
				if featName not in instance.featureSet.featureDict["SyntacticFeatures"]:
					instance.addFeature(self.type, featName, 0.0)

	def get_relation_usage(self, iTrees, instance):
		nTrees = len(iTrees)
		for iTree in iTrees:
			depFreq,_ = iTree.search_deps_frequency()
			for dep, freq in depFreq.items():	
				if "SYNDEP_"+ dep not in instance.featureSet.featureDict["SyntacticFeatures"].keys():
					instance.addFeature(self.type, "SYNDEP_"+dep, 0.0)
				
				instance.updateFeature(self.type, "SYNDEP_"+dep, freq / nTrees)

	def getPassiveUsage(self, iTrees, instance):
		nTrees = len(iTrees)
		instance.addFeature(self.type, "SYNDEP_passive", 0.0)
		for iTree in iTrees:
			depFreq, total = iTree.search_deps_frequency(self.passiveRelations)
			inc = 0
			if total > 0:
				inc = 1

			instance.updateFeature(self.type, "SYNDEP_passive", inc / nTrees)


	def get_relationgroup_usage(self,iTrees, instance):
		nTrees = len(iTrees)
		instance.addFeature(self.type, "SYNDEP_modifierRelations", 0.0)
		instance.addFeature(self.type, "SYNDEP_adverbialRelations", 0.0)

		for iTree in iTrees:
			depFreq, total = iTree.search_deps_frequency(self.adverbialRelations)
			instance.updateFeature(self.type, "SYNDEP_adverbialRelations", total / nTrees)

			depFreq, total = iTree.search_deps_frequency(self.modifierRelations)
			instance.updateFeature(self.type, "SYNDEP_modifierRelations", total / nTrees)


	def get_posgroup_usage(self, iTrees, instance):
		nTrees = len(iTrees)
		instance.addFeature(self.type, "SYNPOS_verbTags", 0.0)
		instance.addFeature(self.type, "SYNPOS_nounTags", 0.0)
		instance.addFeature(self.type, "SYNPOS_adverbTags", 0.0)
		instance.addFeature(self.type, "SYNPOS_adjectiveTags", 0.0)
		instance.addFeature(self.type, "SYNPOS_pronounTags", 0.0)
		instance.addFeature(self.type, "SYNPOS_determinerTags", 0.0)
		instance.addFeature(self.type, "SYNPOS_conjunctionTags", 0.0)
		instance.addFeature(self.type, "SYNPOS_superlatives", 0.0)
		instance.addFeature(self.type, "SYNPOS_comparatives", 0.0)
		instance.addFeature(self.type, "SYNPOS_pastVerbs", 0.0)
		instance.addFeature(self.type, "SYNPOS_presentVerbs", 0.0)


		for iTree in iTrees:
			depFreq, total = iTree.search_pos_frequency(self.verbTags)
			instance.updateFeature(self.type, "SYNPOS_verbTags", total / nTrees)

			depFreq, total = iTree.search_pos_frequency(self.nounTags)
			instance.updateFeature(self.type, "SYNPOS_nounTags", total / nTrees)

			depFreq, total = iTree.search_pos_frequency(self.adverbTags)
			instance.updateFeature(self.type, "SYNPOS_adverbTags", total / nTrees)

			depFreq, total = iTree.search_pos_frequency(self.adjectiveTags)
			instance.updateFeature(self.type, "SYNPOS_adjectiveTags", total / nTrees)

			depFreq, total = iTree.search_pos_frequency(self.pronounTags)
			instance.updateFeature(self.type, "SYNPOS_pronounTags", total / nTrees)

			depFreq, total = iTree.search_pos_frequency(self.determinerTags)
			instance.updateFeature(self.type, "SYNPOS_determinerTags", total / nTrees)

			depFreq, total = iTree.search_pos_frequency(self.conjunctionTags)
			instance.updateFeature(self.type, "SYNPOS_conjunctionTags", total / nTrees)

			depFreq, total = iTree.search_pos_frequency(self.superlatives)
			instance.updateFeature(self.type, "SYNPOS_superlatives", total / nTrees)

			depFreq, total = iTree.search_pos_frequency(self.comparatives)
			instance.updateFeature(self.type, "SYNPOS_comparatives", total / nTrees)

			depFreq, total = iTree.search_pos_frequency(self.pastVerbs)
			instance.updateFeature(self.type, "SYNPOS_pastVerbs", total / nTrees)

			depFreq, total = iTree.search_pos_frequency(self.presentVerbs)
			instance.updateFeature(self.type, "SYNPOS_presentVerbs", total / nTrees)


	def get_pos_usage(self,iTrees, instance):
		nTrees = len(iTrees)
		for iTree in iTrees:
			posFreq, _ = iTree.search_pos_frequency()
			for pos, freq in posFreq.items():
				if "SYNPOS_"+pos not in instance.featureSet.featureDict["SyntacticFeatures"]:
					instance.addFeature(self.type, "SYNPOS_"+pos, 0.0)

				instance.updateFeature(self.type, "SYNPOS_"+pos, freq / nTrees)

	def get_shape_features(self,iTrees, instance):
		nTrees = len(iTrees)
		instance.addFeature(self.type, "SYNSHAPE_width", 0.0)
		instance.addFeature(self.type, "SYNSHAPE_depth", 0.0)
		instance.addFeature(self.type, "SYNSHAPE_ramFactor", 0.0)

		for iTree in iTrees:
			ramFact = iTree.get_ramification_factor()
			width = iTree.get_max_width()
			depth = iTree.get_max_depth()
			instance.updateFeature(self.type, "SYNSHAPE_width", width / nTrees)
			instance.updateFeature(self.type, "SYNSHAPE_depth", depth / nTrees)
			instance.updateFeature(self.type, "SYNSHAPE_ramFactor", ramFact / nTrees)

	def get_verb_features(self, iTrees, instance):
		nTrees = len(iTrees)
		instance.addFeature(self.type, "SYNSHAPE_composedVerbRatio", 0.0)
		instance.addFeature(self.type, "SYNSHAPE_modalRatio", 0.0)

		for iTree in iTrees:
			composedVerbRatio = iTree.get_composed_verb_ratio()
			modalRatio = iTree.get_modal_ratio()
			instance.updateFeature(self.type, "SYNSHAPE_composedVerbRatio", composedVerbRatio / nTrees)
			instance.updateFeature(self.type, "SYNSHAPE_modalRatio", modalRatio / nTrees)
