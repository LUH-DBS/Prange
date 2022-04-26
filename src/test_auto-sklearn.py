# This file is to test if auto-sklearn is installed correctly.

from pprint import pprint

import sklearn.datasets
import sklearn.metrics

import autosklearn.classification

import pandas
X = pandas.read_csv('src/data/training/100-999_50.csv')
y = pandas.read_csv('src/data/training/100-999_50-result.csv')
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, random_state=1)

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    tmp_folder='/tmp/autosklearn_classification_example_tmp',
)
automl.fit(X_train, y_train, dataset_name='breast_cancer')

print(automl.leaderboard())

print("\n-------------------------------\n")

pprint(automl.show_models(), indent=2)

print("\n-------------------------------\n")

predictions = automl.predict(X_test)
print("Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions))

X2 = pandas.read_csv('src/data/training/1000-1499_50_nt.csv')
y2 = pandas.read_csv('src/data/training/1000-1499_50_nt-result.csv')

predictions = automl.predict(X2)
print("Non trivial accuracy score:",
      sklearn.metrics.accuracy_score(y2, predictions))
