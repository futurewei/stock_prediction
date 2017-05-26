import nltk
from nltk.tokenize import word_tokenize
import pickle
from statistics import mode

class voted_classifier:
	def __init__(self, *classifiers):
		self.classifier = classifiers

	def classify(self, data):
		testing_result= []
		for machine in self.classifier:
			testing_result.append(machine.classify(data))
		return mode(testing_result)

	def confidence(self, data):
		testing_result= []
		for machine in self.classifier:
			testing_result.append(machine.classify(data))
		votes = testing_result.count(mode(testing_result))
		return votes/len(testing_result)



open_file = open("algorithm/MNB.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()



open_file = open("algorithm/Ber.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()


open_file = open("algorithm/Logits.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()


open_file = open("algorithm/SGD.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("algorithm/LinearSVC.pickle", "rb")
SVC = pickle.load(open_file)
open_file.close()


voted_classifier = voted_classifier(
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  SGDC_classifier,
                                  SVC)



open_file = open("training_data/data.pickle", "rb")
pos_neg_data = pickle.load(open_file)
open_file.close()



def input_features(documents):
	words = word_tokenize(documents)
	features = {}
	for word in word_tokenize(documents):
			features[word] = (word in words)
	return features

def sentiment(text):
    feats = input_features(text)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)





