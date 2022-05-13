# -*- coding: utf-8 -*-
"""
Created on Fri May 13 10:32:07 2022

@author: Goran
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score



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






def do_corr_filter(data, corr_limit):
    
    initial_list = data.columns.tolist()
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
    
    
    
    
    
    
    
    
    
    