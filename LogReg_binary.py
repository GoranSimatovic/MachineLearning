import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from matplotlib import pyplot as plt
import support_functions as sf

data = pd.read_csv("wine.csv")
data.dropna(inplace=True)

# change labels into 0,1 integers (can be done with OneHot)
data['quality'].replace('bad', 0, inplace=True)
data['quality'].replace('good', 1, inplace=True)

target_variable = 'quality'
list_drivers = data.columns.drop(target_variable).tolist()

target_data = data[target_variable]

#min_max = MinMaxScaler()
standard = StandardScaler()
scaled_data = pd.DataFrame(standard.fit_transform(data[list_drivers]),
                           columns=list_drivers, index=data.index)


# split the data into test/train subsets (no validation)
state_n = 42
test_ratio = 0.2
train_x, test_x = train_test_split(scaled_data,
                                   test_size=test_ratio,
                                   random_state=state_n)

train_y, test_y = train_test_split(target_data,
                                   test_size=test_ratio,
                                   random_state=state_n)


# univariate analysis / feature significance
univariate_results = sf.do_log_univariate(list_drivers,
                                          train_x, test_x,
                                          train_y, test_y)


# pick up only features with gini > 5%
filtered_list = univariate_results\
    .loc[univariate_results.GINI > 0.05]\
    .index.tolist()


# filter out highly correlated features
corr_cutoff = 0.5
filtered_list = sf.do_corr_filter(filtered_list,
                                  scaled_data[filtered_list],
                                  corr_cutoff)


log_clf = LogisticRegression(random_state=0)
log_clf.fit(train_x, train_y)


print('#################\n'
      + 'Full feature set:\n'
      + '#################')
for i in list_drivers:
    print(list_drivers.index(i), i)
print('\nAccuracy:',
      round(accuracy_score(test_y, log_clf.predict(test_x)), 2))
print('GINI score:',
      round(2*roc_auc_score(test_y, log_clf.predict(test_x))-1, 2),
      '\n\n')


log_clf.fit(train_x[filtered_list], train_y)

print('#####################\n'
      + 'Filtered feature set:\n'
      + '#####################')
for i in filtered_list:
    print(filtered_list.index(i), i)
print('\nAccuracy:',
      round(accuracy_score(test_y, log_clf.predict(test_x[filtered_list])), 2))
print('GINI score:',
      round(2*roc_auc_score(test_y,
                            log_clf.predict(test_x[filtered_list]))-1, 2),
      '\n\n')


print('Cool!')


# * Search for best n-tuple of features
# - performance is checked using k-fold cv and GINI

log_clf = LogisticRegression(random_state=0, n_jobs=-1)

best_three = sf.find_best_feature_set(log_clf, train_x[list_drivers],
                                      train_y, 3, list_drivers, 5)

show_first = 10
plt.errorbar(x=range(show_first),
             y=best_three['Gini'][:show_first],
             yerr=best_three['std'][:show_first],
             marker='o', linestyle='--',
             label=f'Best performer - {best_three.index[0]}')

plt.ylim([0.5, 0.7])
plt.xlabel('Features index')
plt.ylabel('Gini [%]')
plt.title('Model performance')
plt.legend()
plt.tight_layout()
plt.show()
