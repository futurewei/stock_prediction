import bs4 as bs 
import requests
import pickle
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from statistics import mean
from Twitter import listener
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
from collections import Counter



def get_sp500_list():
	contents = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
	soup = bs.BeautifulSoup(contents.text, 'html5lib')
	table = soup.find('table', {'class':'wikitable sortable'})
	sp_list = []
	for row in table.findAll('tr')[1:]:
		for col in row.findAll('td')[0]:
			sp_list.append(col.text)

	with open("sp500.pickle","wb") as f:
		pickle.dump(sp_list , f)

	return sp_list

def get_company_data(start, end):
	# when you need to reload

	with open("sp500tickers.pickle", "rb") as f:
		companies = pickle.load(f)

	if not os.path.exists('stock_'):
		os.makedirs('stock_')
	for c in companies[:15]:
		if not os.path.exists('stock_/{}.csv'.format(c)):
			df = web.DataReader(c, "google", start, end)
			df.to_csv('stock_/{}.csv'.format(c))
		if os.path.exists('stock_/{}.csv'.format(c)):
			df = web.DataReader(c, "google", start, end)
			df.to_csv('stock_/{}.csv'.format(c))




def join_data():
	with open("sp500tickers.pickle", "rb") as f:
		companies = pickle.load(f)

	df = pd.DataFrame()
	for company in companies[:15]:
		temp = pd.read_csv('stock_/{}.csv'.format(company))
		temp.set_index('Date', inplace=True)
		temp.rename(columns={'Close':company}, inplace=True)
		temp.drop(['Open','High','Low','Volume'],1, inplace=True)
		if df.empty:
			df = temp
		else:
		    df = df.join(temp, how = 'outer')

	df.to_csv('joined.csv')


#join_data()

def generate_features_and_labels(target, period = 7):
	df = pd.read_csv('joined.csv', index_col=0)
	companies=df.columns.values.tolist()

	df.fillna(0, inplace=True)
	for i in range(1, period+1):
		# percent change
		 df['{}_{}d'.format(target,i)] = (df[target].shift(-i) - df[target]) / df[target]

	df.fillna(0, inplace=True)


	df_vals = df[[c for c in companies]].pct_change()
	df_vals = df_vals.replace([np.inf, -np.inf], 0)
	df_vals.fillna(0, inplace=True)
	features = df_vals.values

	def rise_or_down(*args):
		total =0
		length = len(args)
		for c in args:
			total+=c 
		change = total/length
		if change >0.0075:
			return "rise"
		elif change < -0.075:
			return "down"
		else:
			return "unclear"


	labels = []

	df['target_avg'] = list(map(rise_or_down, df['{}_1d'.format(target)],
		df['{}_2d'.format(target)],
		df['{}_3d'.format(target)],
		df['{}_4d'.format(target)],
		df['{}_5d'.format(target)],
		df['{}_6d'.format(target)],
		df['{}_7d'.format(target)]))
		
	labels = df['target_avg'].values
	return features, labels


def training(target, load = True, retrain=True):
	start = dt.datetime(2010, 5, 1)
	end = dt.datetime(2016, 12, 30)
	get_company_data(start, end)
	join_data()
	features, labels = generate_features_and_labels(target)
	xs, testX, ys, testY = train_test_split(features, labels, test_size=0.25)
	if load == False:
		# initialize new classifier
		clf = svm.LinearSVC()
		clf.fit(xs, ys)
		confidence = clf.score(testX, testY)
		with open("clf.pickle", "wb") as f:
			pickle.dump(clf, f)
	else:
		with open("clf.pickle", "rb") as f:
			clf = pickle.load(f)
		if retrain:
			clf.fit(xs, ys)
		confidence = clf.score(testX, testY)
	print ('accuracy: ', confidence)
	return confidence

'''
with open("sp500tickers.pickle", "rb") as f:
	tickers=pickle.load(f)

accuracy = []
for ticker in tickers[:15]:
	prob = training(ticker)
	accuracy.append(prob)
	print ("accuracy of {} is {}".format(ticker, prob))
	print ("Average is : ", mean(accuracy))
'''

def predict(target):
	start = dt.datetime(2017, 5, 1)
	end = dt.datetime(2017, 5, 15)
	get_company_data(start, end)
	join_data()
	features, labels = generate_features_and_labels(target)
	print (features, labels)
	with open("clf.pickle", "rb") as f:
		clf = pickle.load(f)
	result = clf.predict(features)
	print ("our predicted stock situation: ", result)
	f = open("twitter-out.txt", "r")
	text = f.read().replace('\n', ' ')
	words = text.split()
	distribution = Counter(words)
	length = distribution["neg"] + distribution["pos"]
	neg = distribution["neg"]/length
	pos = distribution["pos"]/length
	if pos > 0.65:
		for i in range(len(result)):
			if result[i] == "down":
				result[i] = "unclear"
			if result[i] =="unclear":
				result[i] = "rise"
		print ("Twitter Sentiment shows that the company is doing good")
	if neg >0.65:
		for j in range(len(result)):
			if result[j] == "rise":
				result[j] = "unclear"
			if result[j] == "unclear":
				result[j] = "down"
		print ("Twitter Sentiment shows that the company is doing not good")
	print ("our adjusted prediction: ")
	print (result)
	return result


def twitter_analysis(target):
	auth = OAuthHandler(ckey, csecret)
	auth.set_access_token(atoken, asecret)
	print ('omg')
	twitterStream = Stream(auth, listener())
	twitterStream.filter(track=[target+' stock'])


predict("MMM")