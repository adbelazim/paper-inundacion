import sklearn
import collections
from sklearn.pipeline import Pipeline
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


import numpy as np

import os
import time

def save_report(path_save, epoch, expected, prediction, mode = "test"):
	path_report = path_save + "/"

	if not os.path.exists(path_report):
		os.makedirs(path_report)

	fname = path_report + "/report_" + str(epoch) + ".txt"
	with open(fname,'w') as f:
		f.write(sklearn.metrics.classification_report(expected, prediction))

# def cross_validation(x_data, y_data, estimator, n_fold, test_size):
# 	scores = []
# 	spliter = StratifiedShuffleSplit(n_splits=n_fold, test_size=test_size, random_state=19993)
# 	for train_index, test_index in spliter.split(x_data, y_data):
# 		x_train = x_data[train_index]
# 		x_test = x_data[test_index]

# 		y_train = y_data[train_index]
# 		y_test = y_data[test_index]

# 		vec_clf.fit(x_train, y_train)
# 		print(vec_clf.score)
# 		scores.append(vec_clf.score)



def build_grid(estimator, param_grid = None, score= 'precision'):
   clf_search = sklearn.model_selection.GridSearchCV(estimator =estimator, \
   								 cv = 5, \
                                 param_grid = param_grid, \
                                 refit = True, \
                                 scoring='%s_weighted' % score)


   return clf_search

def train_cv_grid(X_train, y_train, \
                     path_save, group):
   """Classifier trainer.

   This function is the core of classifier. 
      - First split the tweets data, keeping '20%' of data for validation. 
      - Feature extraction -> Vectorice the data with tf-idf model. Ngram_range = (1,3).
      - Define a classifier. By default SVM.
      - Define a param grid for GridSearchCV procedure. This improve the quality and 
         the generalization of the classifier.
      - Define a list with scores for the GridSearchCV optimization.
      - Pipeline with vectorizer and classifier.
      - Test classifier

   Parameters:

   tweets -- matrix with a tweet in each row.
   labels -- label vector with an label in each row.
   group -- para proposito de guardado de resultados

   """

   # print("y_train",y_train)
   # print("y_test",y_test)

   path_save_svm = path_save + "/SVM"
   if not os.path.exists(path_save_svm):
      os.makedirs(path_save_svm)

   path_save_tree = path_save + "/DTree"
   if not os.path.exists(path_save_tree):
      os.makedirs(path_save_tree)

   path_save_nb = path_save + "/NB"
   if not os.path.exists(path_save_nb):
      os.makedirs(path_save_nb)


   vec = TfidfVectorizer(min_df=5, max_df=0.95, sublinear_tf = True,use_idf = True,ngram_range=(1, 2))

   #SVM classifier
   svm_clf = SVC(C=0.1, probability = True, max_iter = 10000, class_weight='balanced')

   #naive bayes classifier
   nb_clf = MultinomialNB()

   #DecisionTree
   tree_clf = tree.DecisionTreeClassifier(class_weight = 'balanced')

   #Define parameter for the gridsearch. This grid include linear SVM.
   svm_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1,1e-2,1e-3, 1e-4,1,10,100,1000],
                     'C': [0.01,0.1,1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.01,0.1,1, 10, 100, 1000]}]

   tree_parameters = {'criterion' : ['gini','entropy']}

   nb_parameters = {'alpha': [1.0, 0.0]}
 
   #models = {'tree' : tree_clf,
   #         'svm' : svm_clf}
   models = {'nb' : nb_clf}
   #define metrics to optimize the gridsearch. Recall must to be better
   scores = ['precision', 'recall','f1']
   
   best_scores_svm = {}
   best_scores_dt = {}
   best_scores_nb = {}

   for model in models:
      clf = models.get(model) 
      #gridsearch must to be independent for every score.
      for score in scores:
         #grid search definition
         if model == 'svm':
            #print("Train SVM for score: ", score)
            path_save = path_save_svm
            #param grid
            grid_parameters = svm_parameters

            #buidl grid search
            clf_search = build_grid(clf,grid_parameters,score)
            vec_clf = Pipeline([('vectorizer', vec), ('pac', clf_search)])

            #fit the classifier to the train data
            vec_clf.fit(X_train, y_train)
            #print("results",clf_search.best_score_)
            best_scores_svm[score] = clf_search.best_score_

            #file to save results
            dir_save_pkl = path_save + '/Model/'
            path_save_pkl = dir_save_pkl + score + '_' + model + '_' + str(group) +'classifier.pkl'
            if not os.path.exists(dir_save_pkl):
               os.makedirs(dir_save_pkl)

            dir_save_report = path_save + '/Report/'
            path_save_report = dir_save_report + score
            if not os.path.exists(dir_save_report):
               os.makedirs(dir_save_report)

            joblib.dump(clf_search, path_save_pkl, compress=3)

            #print("SVM trained for score: ", score)

         elif model == 'nb':
            #print("Train SVM for score: ", score)
            path_save = path_save_nb
            #param grid
            grid_parameters = nb_parameters

            #buidl grid search
            clf_search = build_grid(clf,grid_parameters,score)
            vec_clf = Pipeline([('vectorizer', vec), ('pac', clf_search)])

            #fit the classifier to the train data
            vec_clf.fit(X_train, y_train)
            #print("results",clf_search.best_score_)
            best_scores_nb[score] = clf_search.best_score_

            #file to save results
            dir_save_pkl = path_save + '/Model/'
            path_save_pkl = dir_save_pkl + score + '_' + model + '_' + str(group) +'classifier.pkl'
            if not os.path.exists(dir_save_pkl):
               os.makedirs(dir_save_pkl)

            dir_save_report = path_save + '/Report/'
            path_save_report = dir_save_report + score
            if not os.path.exists(dir_save_report):
               os.makedirs(dir_save_report)

            joblib.dump(clf_search, path_save_pkl, compress=3)

            #print("SVM trained for score: ", score)

         else:
            #print("Train DTree for score: ", score)
            path_save = path_save_tree
            #param grid
            grid_parameters = tree_parameters

            #buidl grid search
            clf_search = build_grid(clf,grid_parameters,score)
            vec_clf = Pipeline([('vectorizer', vec), ('pac', clf_search)])

            #fit the classifier to the train data
            vec_clf.fit(X_train, y_train)
            #print("results",clf_search.best_score_)
            best_scores_dt[score] = clf_search.best_score_

            #file to save results
            dir_save_pkl = path_save + '/Model/'
            path_save_pkl = dir_save_pkl + score + '_' + model + '_' + str(group) + 'classifier.pkl'
            if not os.path.exists(dir_save_pkl):
               os.makedirs(dir_save_pkl)

            dir_save_report = path_save + '/Report/'
            path_save_report = dir_save_report + score 
            if not os.path.exists(dir_save_report):
               os.makedirs(dir_save_report)

            #print("Dtree trained for score: ", score)

   #print("best_scores_svm",best_scores_svm)
   #print("best_scores_dt",best_scores_dt)
   print("best_scores_dt",best_scores_nb)
   

   #return best_scores_svm, best_scores_dt
   return best_scores_nb
         

         #predictions for
         #train_predict = cross_val_predict(clf_search, X_train, y_train, cv = cv)
         # train_predict = vec_clf.predict(X_train)
         # test_predict = vec_clf.predict(X_test)

         # #predictions probs for roc
         # train_predict_proba = vec_clf.predict_proba(X_train)
         # test_predict_proba = vec_clf.predict_proba(X_test)

         # save_report(path_save_report, group, y_test, test_predict, mode = "test")





