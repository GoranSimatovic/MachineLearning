import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, roc_auc_score

import support_functions as sf

data = pd.read_csv("wine.csv")
data.dropna(inplace=True)

#change labels into 0,1 integers (can be done with OneHot)
data['quality'].replace('bad',0,inplace=True)
data['quality'].replace('good',1,inplace=True)

target_variable = 'quality'
list_drivers = data.columns.drop(target_variable).tolist()

target_data = data[target_variable]

#min_max = MinMaxScaler()
standard = StandardScaler()
scaled_data = pd.DataFrame(standard.fit_transform(data[list_drivers]),
                           columns = list_drivers, index = data.index)


target_data = data[target_variable]



# split the data into test/train subsets (no validation)
train_x, test_x = train_test_split(scaled_data,
                                   test_size=0.2,
                                   random_state=42)

train_y, test_y = train_test_split(target_data,
                                   test_size=0.2,
                                   random_state=42)



#univariate analysis / feature significance
univariate_results = sf.do_log_univariate(list_drivers,
                                          train_x, test_x, 
                                          train_y, test_y)
                                          


filtered_list = univariate_results\
                .loc[univariate_results.GINI>0.05]\
                .index.tolist()



#filter out highly correlated features
corr_cutoff = 0.5
filtered_list = sf.do_corr_filter(scaled_data[filtered_list], corr_cutoff)
    


#define several learning methodologies with different logic
log_clf = LogisticRegression(random_state=0)




log_clf.fit(train_x, train_y)

print('Full feature set:')
print(list_drivers)
print('Accuracy:',
      round(accuracy_score(test_y, log_clf.predict(test_x)),2))
print('GINI score:',
      round(2*roc_auc_score(test_y, log_clf.predict(test_x))-1,2),
      '\n\n')


log_clf.fit(train_x[filtered_list], train_y)

print('Filtered feature set:')
print('Accuracy:',
      round(accuracy_score(test_y, log_clf.predict(test_x[filtered_list])),2))
print('GINI score:',
      round(2*roc_auc_score(test_y, log_clf.predict(test_x[filtered_list]))-1,2))



