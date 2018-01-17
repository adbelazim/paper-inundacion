# -*- coding: utf-8 -*-

from sklearn.externals import joblib
import sys
import process_tweet as pt
import argparse

def predict(tweet,classifier):
	tweet_processed = pt.stem(pt.process_tweet(tweet))
	#print("tweet_processed",tweet_processed)
	X =  [tweet_processed]
	sentiment = classifier.predict(X)

	return (sentiment[0])


def load_classifier():
	print('Loading the Classifier, please wait....')
	classifier = joblib.load('svmClassifier.pkl')
	print('READY')
	return classifier

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-t", "--tweet", help="Tweet to classify")
	args = parser.parse_args()
	classifier = load_classifier()
	print(predict(str(args.tweet),classifier))
	#tweet = ' '
	#for tweet in sys.stdin:
	#	print(predict(tweet, classifier))

















