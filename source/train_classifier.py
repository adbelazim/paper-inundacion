import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import numpy as np

import process_tweet
import data
import data_split

import os

#this function train and generate report of the classifier
def classifier(X_train,y_train,X_test,y_test):
   vec = TfidfVectorizer(min_df=5, max_df=0.95, sublinear_tf = True,use_idf = True,ngram_range=(1, 2))
   svm_clf =svm.LinearSVC(C=0.1)
   vec_clf = Pipeline([('vectorizer', vec), ('pac', svm_clf)])
   vec_clf.fit(X_train,y_train)
   joblib.dump(vec_clf, 'svmClassifier.pkl', compress=3)
   y_pred = vec_clf.predict(X_test)
   with open("report.txt",'w') as f:
      f.write(sklearn.metrics.classification_report(y_test, y_pred))


def main():
   path = os.getcwd()
   path_save, path_source = os.path.split(path)
   path_save = path_save + "/Results"
   if not os.path.exists(path_save):
      os.makedirs(path_save)
   
   #tweets,labels=data.read_data("../data/dataset_terremoto_iquique_2014.csv")
   #tweets,labels = data.read_data("../data/tweets-iquique-2014-tipo-informacion.csv")
   tweets_data = data.read_data("../data/results-random-inf.csv")
   #print(tweets_data[13])

   #group = 0
   for tweet_info in tweets_data:
      processed_tweets = process_tweet.process_tweets(tweets_data[tweet_info]["tweets"])


   #ACA EMPIEZA EL PROCESO DE "ACTIVE LEARNING"

   #se entrena el grupo 0 solo inicialmente, luego dentro del loop se entrena el resto de los grupos aucmulados
   print("entrenando grupo 0")
   #best_score_svm, best_score_dt = data_split.train_cv_grid(tweets_data[0]["tweets"], tweets_data[0]["labels"], \
   #            path_save, 0)
   best_score_nb = data_split.train_cv_grid(tweets_data[0]["tweets"], tweets_data[0]["labels"], \
               path_save, 0)

   tweets_data_new = {}
   dict_new = {}
   
   for group in tweets_data:
      for key in tweets_data[0].keys():
         if group == 0:
            dict_new[key] = tweets_data[group][key] + tweets_data[group+1][key]
            tweets_data_new[group] = dict_new
         else:
            if (group + 1) in tweets_data:
               dict_new[key] = dict_new[key] + tweets_data[group+1][key]
               tweets_data_new[group] = dict_new
      #print(dict_new)
      #X_train, X_test, y_train, y_test = train_test_split(dict_new["tweets"],dict_new["labels"], stratify = dict_new["labels"], \
      #                                                         test_size=0.20, random_state=199993)
      if group > 0:
         print("entrenando con grupo: ", group)
         #best_score_svm, best_score_dt = data_split.train_cv_grid(dict_new["tweets"], dict_new["labels"], \
         #   path_save, group)
         best_score_nb = data_split.train_cv_grid(dict_new["tweets"], dict_new["labels"], \
            path_save, group)

         
      #dict_new = {}

   #print(tweets_data_new[0])

   #con esto se prueba para el grupo 0 las metricas que se obtienen.
   # X_train, X_test, y_train, y_test = train_test_split(tweets_data[0]["tweets"],tweets_data[0]["labels"], stratify = tweets_data[0]["labels"], \
   #                                                           test_size=0.20, random_state=42)
   # #con esto se entrena cross validation y grid search una SVM y Arboles de decision
   # data_split.train_cv_grid(X_train, y_train, \
   #   X_test, y_test, \
   #   path_save, 0)
      
  


   #classifier(X_train,y_train,X_test, y_test)

if __name__=="__main__":
   main()












