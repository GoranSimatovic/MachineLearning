# -*- coding: utf-8 -*-
"""
Created on Fri May 13 10:32:07 2022

@author: Goran
"""

import numpy as np
import pandas as pd
import time
from itertools import combinations
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score



###
# * Logistic regression performed on each feature independently
# * Features scored using GINI (2*roc_AUC-1)
# * Input: list of features,pd.DataFrame format of train/test x/y
# * Output: pd.DataFrame with column 'GINI'
###
def do_log_univariate(list_drivers, train_x, test_x, train_y, test_y):
    
    #list of results
    univariate_gini_scores = pd.DataFrame(columns = ['GINI'], index = list_drivers)
    
    log_clf = LogisticRegression()

    
    #loop over all drivers, fit and predict (test/train)
    #results are written into a pd.DataFrame
    for driver in list_drivers:
        log_clf.fit(train_x[driver].to_numpy().reshape(-1, 1), train_y)
        univariate_gini_scores.loc[driver,'GINI'] = \
            2*roc_auc_score(test_y,log_clf.predict(test_x[driver]\
                                          .to_numpy()\
                                          .reshape(-1, 1)))-1
        
    #return sorted scores
    return univariate_gini_scores.sort_values('GINI', ascending=False)



###
# * Downward filtering of less performing features with respect to correlation
# - code will loop the descending performance list of features and check for 
#   features that are overcorrelated with a higher performing feature and remove
#   any such feature
#
# Input: initial_list -> list with features in descending order in performance,
#        data -> pd.DataFrame with at least initial_list columns
#        corr_limit -> upper limit for correlation cut-off
# Output: list of ordered and filtered features
###


def do_corr_filter(initial_list, data, corr_limit):
    
    corr_df = data.corr()    
    
    filtered_list = [] 
    while len(initial_list)>1:
        investigated_feature = initial_list[0]
        for feature in initial_list[1:]:
            if np.abs(corr_df.loc[investigated_feature, feature])>corr_limit:
                initial_list.remove(feature)
        filtered_list.append(investigated_feature)
        initial_list.remove(investigated_feature)
        
    return filtered_list
    
    
    
###
# * Code for finding the best performing set of n features  
# - Code will use proposed sklearn classifier and the input data to look
#   for best n features, using k-fold cross-validation and gini scores
# Input: classifier -> sklearn model (LogisticRegression, RandomForestClassifier)
#        train_data_x, train_data_y -> usual pd.DataFrame with input_list features
#                                      and the target variable
#        n -> integer number of features to be used for input
#        input_list -> list of all considered features
#        n_k_fold -> integer number of folds to be used with KFold
# Output: pd.DataFrame ordered with columns 'Gini' and 'std' (std. dev.)
###
def find_best_feature_set(classifier, train_data_x, train_data_y,\
                          n, input_list, n_k_fold):
    
    print('\nLooking for a best set of %i features\n' %n)
    cv = KFold(n_splits = n_k_fold, random_state = 42, shuffle = True)
    set_of_feature_lists = list(combinations(input_list,n))
    size_of_tuple_list = len(set_of_feature_lists)
    scores = pd.DataFrame(columns = ['Gini', 'std'],\
                            index = set_of_feature_lists)
    
    time_start = time.time()
    iteration_intervals = 10
    n_progress_stars = 30
    for i, feature_selection in zip(range(1,size_of_tuple_list+1),\
                                    set_of_feature_lists):
        
        if i%iteration_intervals==iteration_intervals-1:
            time_now = time.time()
            
            progress_count = round(n_progress_stars*i/size_of_tuple_list)
            print('\nProgress: <', '*'*progress_count + \
                  'x'*(n_progress_stars-progress_count)+' >')
            print('Passed time: %s'\
                  % str(round(time_now-time_start,2)) +'s\n')
        
        feature_selection = list(feature_selection)
        results = cross_val_score(classifier, train_data_x[feature_selection],\
                                  train_data_y, scoring = 'roc_auc', cv = cv, \
                                  n_jobs = -1)
        
        scores.loc[str(feature_selection), 'Gini'] = np.mean(results)*2.0-1.0
        scores.loc[str(feature_selection), 'std'] = np.std(results*2.0-1.0)
    
    scores
    return scores.sort_values(by='Gini', ascending = False).dropna()

    
    
    
    
    