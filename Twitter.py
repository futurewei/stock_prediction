from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s





class listener(StreamListener):
	def on_data(self, data):
		all_data = json.loads(data)
		try:
			tweet = all_data["text"]
			sentiment_value, confidence = s.sentiment(tweet)
			if confidence*100 >=80:
				output = open("twitter-out.txt", "a")
				output.write(sentiment_value)
				output.write('\n')
				output.close()
			else:
				return True
		except:
			print ("Error tweet")
		return True

	def on_error(self, status):
		print (status)



'''
sample run on "Free speech"

RT @JohnFromCranber: What Libs Are Protesting Against is The Free Speech of Folks Who Have  
Views That Differ From Theirs.... They're NOT L… pos 0.6
RT @KanchanGupta: Blame it on official denial of free speech. Plus the urge to be politically 
correct. We were PC long before world d…  neg 1.0
RT @RepJoeKennedy: In this country we defend both the right to free speech and those who exercise it. 
https://t.co/kCiv3TXoLa pos 0.6
Our Views: Protecting free speech doesn't require new laws https://t.co/hWJmHIRWii neg 0.8

'''