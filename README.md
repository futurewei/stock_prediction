# stock_predictor

The stock_predictor is implemented by Machine Learning and Sentiment Analysis over twitter comments about a certain company. I trained my SVM classifier using data from 2000-2016 stock price as training data.
The feature I used is the percent change between SP500 companies and seek the influence on my target company. [This idea is inspired by sentex]. The machine learning method alone is able to achieve an accuracy about 60%. To improve my accuracy,
I also use Twitter's comments about a company as reference. Unforntunately, I don't have historical Twitter data as my testing set, so I only use it as a reference hoping to improve my result.
