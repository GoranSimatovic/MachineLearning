# -*- coding: utf-8 -*-
"""
Created on Sat May 21 18:20:39 2022

@author: Goran
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score
from matplotlib import pyplot as plt
import support_functions as sf

data = pd.read_csv("wine.csv")
data.dropna(inplace=True)

#change labels into 0,1 integers (can be done with OneHot)
data['quality'].replace('bad',0,inplace=True)
data['quality'].replace('good',1,inplace=True)

target_variable = 'quality'
list_drivers = data.columns.drop(target_variable).tolist()

target_data = data[target_variable]



# split the data into test/train subsets (no validation)
state_n = 42
test_ratio = 0.2
train_x, test_x = train_test_split(data[list_drivers],
                                   test_size=test_ratio,
                                   random_state=state_n)

train_y, test_y = train_test_split(target_data,
                                   test_size=test_ratio,
                                   random_state=state_n)



# Feature significance
                                         
RF_clf = RandomForestClassifier(n_jobs = -1,
                                n_estimators=100,
                                random_state = 23)

RF_clf.fit(train_x, train_y)

results_stack = pd.DataFrame(
                {'Feature Importance':RF_clf.feature_importances_},
                index = list_drivers).sort_values('Feature Importance',
                                                  ascending=False)

# Measure performance for models with devreasing number of drivers,
# each new iteration has the least performing feature/s removed
features = list_drivers
backward_scores = pd.DataFrame(columns = ['gini', 'feature_list'],
                               index = range(3,len(features)))
while len(features) > 2:
    RF_clf.fit(train_x[features], train_y)
    gini = 2*roc_auc_score(test_y,RF_clf.predict(test_x[features]))-1

    n_features = len(features)
    backward_scores.loc[n_features,
                        'gini'] = gini
    backward_scores.loc[n_features,
                        'feature_list'] = ''.join([x+'__' for x in features])

    importance_df = pd.DataFrame({'features':features,
                                  'importance':RF_clf.feature_importances_})
    least_importance = importance_df.importance.min()
    features = importance_df.loc[importance_df.importance>least_importance,
                                 'features']


print(backward_scores)
max_score = backward_scores.gini.max()
final_list_via_importance =  backward_scores\
                             .loc[backward_scores.gini == max_score,\
                             'feature_list'].values[0].split('__')[:-1]





#Compare full feature list and the filtered set

RF_clf = RandomForestClassifier(n_jobs = -1,
                                n_estimators=100,
                                random_state = 23)

RF_clf.fit(train_x, train_y)
print('#################\n'\
+     'Full feature set:\n'\
+     '#################')
for i in list_drivers:
    print(list_drivers.index(i), i)
print('\nAccuracy:',
      round(accuracy_score(test_y, RF_clf.predict(test_x)),2))
print('GINI score:',
      round(2*roc_auc_score(test_y, RF_clf.predict(test_x))-1,2),
      '\n\n')


RF_clf.fit(train_x[final_list_via_importance], train_y)

print('#####################\n'\
+     'Filtered feature set:\n'\
+     '#####################')
for i in final_list_via_importance:
    print(final_list_via_importance.index(i), i)
print('\nAccuracy:',
      round(accuracy_score(test_y, RF_clf.predict\
                          (test_x[final_list_via_importance])),2))
print('GINI score:',
      round(2*roc_auc_score(test_y, RF_clf.predict\
                           (test_x[final_list_via_importance]))-1,2),
      '\n\n')


print('Cool!')