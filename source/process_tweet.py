import re
import nltk

def stem(tweet):
   stemmer = nltk.stem.PorterStemmer()
   tweet_stem = ''
   words = [word if(word[0:2]=='__') else word.lower() \
      for word in tweet.split() \
      if len(word) >= 3]
   words = [stemmer.stem(w) for w in words] 
   tweet_stem = ' '.join(words)
   return tweet_stem

def process_tweets(tweets):
   processed_tweets = [stem(process_tweet(tweet)) for tweet in tweets]
   return processed_tweets   

def process_tweet(tweet):
   #Convert www.* or https?://* to URL
   tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    
   #Convert @username to __HANDLE
   tweet = re.sub('@[^\s]+','__HANDLE',tweet)  
    
   #Replace #word with word
   tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

   #trim
   tweet = tweet.strip('\'"')
    
   # Repeating words like happyyyyyyyy
   rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE)
   tweet = rpt_regex.sub(r"\1\1", tweet)
    
   #Emoticons
   emoticons = \
   [
   ('__positive__',[ ':-)', ':)', '(:', '(-:', \
                     ':-D', ':D', 'X-D', 'XD', 'xD', \
                     '<3', ':\*', ';-)', ';)', ';-D', ';D', '(;', '(-;', ] ),\
   ('__negative__', [':-(', ':(', '(:', '(-:', ':,(',\
                     ':\'(', ':"(', ':((', ] ),\
   ]

   def replace_parenth(arr):
      return [text.replace(')', '[)}\]]').replace('(', '[({\[]') for text in arr]

   def regex_join(arr):
      return '(' + '|'.join( arr ) + ')'

   emoticons_regex = [ (repl, re.compile(regex_join(replace_parenth(regx))) ) \
      for (repl, regx) in emoticons ]

   for (repl, regx) in emoticons_regex :
      tweet = re.sub(regx, ' '+repl+' ', tweet)

   #Convert to lower case
   tweet = tweet.lower()

   return tweet







