import pandas as pd
from matplotlib import pyplot

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve


titanic_data = pd.read_csv("titanic.csv")

# organize the dataset
titanic_data = titanic_data.drop(['Name'], axis=1)
#change sex labels into 0,1 integers
titanic_data['Sex'].replace('female',0,inplace=True)
titanic_data['Sex'].replace('male',1,inplace=True)
# split the data into test/train subsets (no validation)
train_set, test_set = train_test_split(titanic_data, test_size=0.2, random_state=42)

# separate outcome labels
train_set_labels = train_set['Survived']
test_set_labels = test_set['Survived']
# remove outcome labels
train_set = train_set.drop('Survived',axis=1)
test_set = test_set.drop('Survived',axis=1)

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC(probability=True)

voting_clf = VotingClassifier(
estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
voting='soft')

for clf in (log_clf,  rnd_clf, svm_clf, voting_clf):
    clf.fit(train_set, train_set_labels)
    fpr, tpr, _ = roc_curve(test_set_labels, clf.predict_proba(test_set)[:, 1])
    pyplot.plot(fpr, tpr, marker='.', label=clf.__class__.__name__)
    print(clf.__class__.__name__, accuracy_score(test_set_labels, clf.predict(test_set)))


pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show()