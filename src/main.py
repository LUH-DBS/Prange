import os
from pprint import pprint
from dotenv import load_dotenv
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score
from genericpath import exists
import pickle

import datasets.local as local
from datasets.sql import Gittables, Maintables, OpenData
import datasets.sql.csv_cache as csv_cache
import algorithms.naiveAlgorithm as naiveAlgorithm
import algorithms.machineLearning as machineLearning

import testing

load_dotenv()
db_params = {
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
}


def main():
    # if exists('test.pickle'):
    #     with open('test.pickle', 'rb') as file:
    #         ml = pickle.load(file)
    # else:
    #     ml = machineLearning.train(
    #         "src/data/training/100-999_10.csv", "test.pickle")
    # X2 = pd.read_csv('src/data/training/1000-1499_10_nt.csv').values
    # y2 = pd.read_csv('src/data/training/1000-1499_10_nt-result.csv').values

    # predictions = ml.predict(X2)
    # print("Non trivial accuracy score:", accuracy_score(y2, predictions))
    # print("Non trivial precision score:", precision_score(y2, predictions))
    testing.prepare_and_train(
        [5, 10], 100, 100, 'src/data/gittables/object_tables')


if __name__ == '__main__':
    main()
