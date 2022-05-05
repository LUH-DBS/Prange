# This file is to test if auto-sklearn is installed correctly.

from pprint import pprint

import sklearn.datasets
import sklearn.metrics

import autosklearn.classification

import pandas


def main():
    X = pandas.read_csv('src/data/training/100-999_5.csv')
    y = pandas.read_csv('src/data/training/100-999_5-result.csv')

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, random_state=1)

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        tmp_folder='/tmp/autosklearn_classification_example_tmp',
    )
    automl.fit(X_train, y_train, dataset_name="Test")

    # print(automl.leaderboard())
    # print("\n-------------------------------\n")
    # pprint(automl.show_models(), indent=2)
    # print("\n-------------------------------\n")

    predictions = automl.predict(X_test)
    print("Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions))

    X2 = pandas.read_csv('src/data/training/1000-1499_5_nt.csv').values
    y2 = pandas.read_csv('src/data/training/1000-1499_5_nt-result.csv').values

    predictions = automl.predict(X2)
    print("Non trivial accuracy score:",
          sklearn.metrics.accuracy_score(y2, predictions))

    save_debug_csv(X2, predictions, y2)


def save_debug_csv(dataset, predictions, ground_truth):
    # for row_index, (input, prediction, label) in enumerate(zip(X2, predictions, y2)):
    #     if prediction != label:
    #         print('Row', row_index, 'has been classified as',
    #               prediction, 'and should be', label, '--', input)
    debug_false_rows = pandas.DataFrame([])
    debug_correct_rows = pandas.DataFrame([])
    for input, prediction, label in zip(dataset, predictions, ground_truth):
        if prediction != label:
            # print(input, 'has been classified as', prediction, 'and should be', label)
            debug_false_rows = pandas.concat(
                [debug_false_rows, pandas.DataFrame(input).T])
        else:
            debug_correct_rows = pandas.concat(
                [debug_correct_rows, pandas.DataFrame(input).T])
    debug_false_rows.to_csv('debug_false_rows.csv', index=False, header=False)
    debug_correct_rows.to_csv('debug_correct_rows.csv',
                              index=False, header=False)


if __name__ == "__main__":
    main()
