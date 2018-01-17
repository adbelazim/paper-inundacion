from nltk.stem.snowball import SnowballStemmer
import nltk


def get_stopwords():
   stopwords = nltk.corpus.stopwords.words('spanish')
   return stopwords

def get_stemm():
   stemmer = SnowballStemmer("spanish")
   return stemmer

def tokenize_and_stem(tweet):
   stemmer = get_stemm()
   # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
   tokens = [word for sent in nltk.sent_tokenize(tweet) for word in nltk.word_tokenize(sent)]
   filtered_tokens = []
   # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
   for token in tokens:
      if re.search('[a-zA-Z]', token):
         filtered_tokens.append(token)
   stems = [stemmer.stem(t) for t in filtered_tokens]
   return stems