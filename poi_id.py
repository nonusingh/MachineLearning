#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','salary','to_messages', 'deferral_payments',
                'total_payments','exercised_stock_options','bonus','restricted_stock',
                'shared_receipt_with_poi','total_stock_value','expenses','from_messages',
                'other','from_this_person_to_poi','deferred_income','long_term_incentive',
                'from_poi_to_this_person']# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print "length of the dataset:"
print len(data_dict)

#  What is the Number of features in the dict?
unique_features = set(
    feature
    for row_dict in data_dict.values()
    for feature in row_dict.keys()
)
print "number of unique features:"
print(len(unique_features))
print(unique_features)

# How many POIs in the dataset? How many are not POIs
count = 0
for user in data_dict:
    if data_dict[user]['poi'] == True:
        count+=1
print "number of employees that are 'persons of interest':"
print (count)

print "number of employees that are not POI:"
print len(data_dict)-(count)

### Task 2: Remove outliers
import matplotlib.pyplot
data = featureFormat(data_dict, features_list)
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

# How to handle outlier with 1e7 bonus?
data_dict.pop('TOTAL', 0)

# How many features have most missing values?
missing_features = {}
for feature in unique_features:
    missing_count = 0
    for k in data_dict.iterkeys():
        if data_dict[k][feature] == "NaN":
            missing_count += 1
    missing_features[feature] = missing_count
print missing_features

### Task 3: Create new feature(s)
# New Feature: from_poi
for employee in data_dict:
    if (data_dict[employee]['to_messages'] not in ['NaN', 0]) and (data_dict[employee]['from_this_person_to_poi'] not in ['NaN', 0]):
        data_dict[employee]['from_poi'] = float(data_dict[employee]['to_messages'])/float(data_dict[employee]['from_this_person_to_poi'])
    else:
        data_dict[employee]['from_poi'] = 0

### Store to my_dataset for easy export below.
my_dataset = data_dict
features_list = ['poi','salary','to_messages', 'deferral_payments',
                'total_payments','exercised_stock_options','bonus','restricted_stock',
                'shared_receipt_with_poi','total_stock_value','expenses','from_messages',
                'other','from_this_person_to_poi','deferred_income','long_term_incentive',
                'from_poi_to_this_person','from_poi']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn import cross_validation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA

#algo = AdaBoostClassifier(algorithm='SAMME')
#scaler =MinMaxScaler()
#skb = SelectKBest(k = 10)
sss = StratifiedShuffleSplit(n_splits= 100, test_size= 0.3, random_state= 42)
pca = PCA()
select = SelectKBest()
#pca = PCA(n_components = 3)


# Provided to give you a starting point. Try a variety of classifiers.


scaler = MinMaxScaler()
#skb = SelectKBest()
abc = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_features='sqrt',min_samples_split=3,min_samples_leaf=2),algorithm='SAMME')

param_grid = {
        'abc__n_estimators':[10,30,50,70,100,150,200],
        'abc__learning_rate':[0.1,0.25,0.5,0.75,1.0,1.5],
        'pca__n_components':[1, 2, 3]
         }

pipeline = Pipeline(steps=[('scaler', scaler), ('pca', pca), ('abc', abc)])

gs= GridSearchCV(pipeline, param_grid, cv = sss, scoring = 'f1')


#parameters = {"base_estimator__criterion" : ["gini", "entropy"],
              #"base_estimator__splitter" :   ["best", "random"],
              #"n_estimators": [1, 10, 50, 100, 200], 
              #'learning_rate':[0.1,0.2,0.25,0.3,0.5,0.75,1.0]
             #}


#DTC = tree.DecisionTreeClassifier(random_state = 60, max_features = "auto", class_weight = "balanced", max_depth = None)

#Adaboost_tuned = AdaBoostClassifier(base_estimator = DTC)

# run grid search
#gs_ABC = GridSearchCV(Adaboost_tuned, param_grid=parameters) 

gs.fit(features_train, labels_train)
print "The best parameters for the grid:"
print(gs.best_params_)
print ' '

#pipe =  Pipeline(steps=[('scaling',scaler),('skb', skb), ("pca", pca), ("Tree", algo)])

clf = gs.best_estimator_
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print "Precision score is:", precision_score(labels_test, pred)
print "Recall score is: ", recall_score(labels_test, pred)
print "-------------------"
print "Classification Report:\n ", classification_report(labels_test, pred)
print "-------------------"
print "Confusion Matrix:\n ", confusion_matrix(labels_test, pred)
print "-------------------"

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)