import pandas as pd
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score

import support_functions as sf

titanic_data = pd.read_csv("titanic.csv")

list_drivers = titanic_data.columns
list_drivers = list_drivers.drop(['Name','Survived'])

# remove private names
titanic_data = titanic_data.drop(['Name'], axis=1)

#change sex labels into 0,1 integers (can be done with where but less efficient)
titanic_data['Sex'].replace('female',0,inplace=True)
titanic_data['Sex'].replace('male',1,inplace=True)

standard = StandardScaler()

scaled_data = pd.DataFrame(standard.fit_transform(titanic_data[list_drivers]),
                           columns = list_drivers, index = titanic_data.index)
target_data = titanic_data.Survived


# split the data into test/train subsets (no validation)
train_set, test_set = train_test_split(scaled_data, test_size=0.2,
                                       random_state=42)

train_set_labels, test_set_labels = train_test_split(target_data, test_size=0.2,
                                                     random_state=42)


#univariate analysis / feature significance
univariate_results = sf.do_log_univariate(list_drivers, train_set, test_set, 
                                          train_set_labels, test_set_labels)
                                          

filtered_list = univariate_results.loc[univariate_results.GINI>0.05]\
                                  .index.tolist()


#define several learning methodologies with different logic
log_clf = LogisticRegression(random_state=0)
rnd_clf = RandomForestClassifier()
svm_clf = SVC(probability=True)
voting_clf = VotingClassifier(
estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
voting='soft')

for clf in (log_clf,  rnd_clf, svm_clf, voting_clf):
    
    clf.fit(train_set[filtered_list], train_set_labels)
    fpr, tpr, _ = roc_curve(test_set_labels,
                            clf.predict_proba(test_set[filtered_list])[:, 1])
    
    pyplot.plot(fpr, tpr, marker='.', label=clf.__class__.__name__)
    print('\n', clf.__class__.__name__)
    print('Accuracy:', 
          round(accuracy_score(test_set_labels, 
                               clf.predict(test_set[filtered_list])),2))
    print('GINI score:', 
          round(2*roc_auc_score(test_set_labels, 
                                clf.predict(test_set[filtered_list]))-1,2))

    
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend()
    pyplot.show()
    

