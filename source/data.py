import nltk
import csv
from bs4 import BeautifulSoup


# def read_data(file_data):
#    """
#    Lectura de archivo .csv que retorna una lista de tweets y labels.

#    Parametros:
#    file_data -- ruta 

#    Excepciones:

   
#    """
#    labels = []
#    tweets = []

#    with open(file_data,'r') as f:
#       for line in f:
#          stream_line = line.split('\t')
#          labels.append(stream_line[0].decode('iso-8859-1').encode('utf8'))
#          tweets.append(stream_line[1].decode('iso-8859-1').encode('utf8'))

#    tweets = clean_data(tweets)
#    return tweets, labels


def read_data(file_data):
   """
   Lectura de archivo .csv que retorna una lista de tweets y labels.

   Parametros:
   file_data -- ruta 

   Excepciones:

   
   """
   labels = []
   tweets = []
   tweets_id = []
   groups = []

   training_data = {}
   tweet_dict = {}

   with open(file_data,'r') as f:
      aux_group = 0
      for line in f:
         stream_line = line.split('\t')

         if str(aux_group) != stream_line[3]:
            #print("entre")
            #print(aux_group)
            tweets = clean_data(tweets)
            tweet_dict["tweets_id"] = tweets_id
            tweet_dict["tweets"] = tweets
            tweet_dict["labels"] = labels
            tweet_dict["groups"] = groups
            training_data[aux_group] = tweet_dict
            labels = []
            tweets = []
            tweets_id = []
            groups = [] 
            tweet_dict = {}
            aux_group = int(stream_line[3])

         #0 tweet id
         tweets_id.append(stream_line[0])
         #2 category
         labels.append(stream_line[2].decode('iso-8859-1').encode('utf8'))
         #3 group
         groups.append(stream_line[3])
         #4 tweet
         tweets.append(stream_line[4].decode('iso-8859-1').encode('utf8'))

      tweets = clean_data(tweets)
      tweet_dict["tweets_id"] = tweets_id
      tweet_dict["tweets"] = tweets
      tweet_dict["labels"] = labels
      tweet_dict["groups"] = groups
      training_data[aux_group] = tweet_dict
   #tweets = clean_data(tweets)

   #data = zip(tweets_id, tweets, labels, groups)

   #return tweets, labels, groups, tweets_id
   return training_data


def clean_data(data):
   clean_uni_data = []
   for text in data:
      text = BeautifulSoup(text, 'html.parser').getText()
      #strips html formatting and converts to unicode
      clean_uni_data.append(text)
   return clean_uni_data










