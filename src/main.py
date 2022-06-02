import os
from dotenv import load_dotenv
import pandas as pd

from autosklearn.metrics import accuracy, precision, recall, f1

import datasets.local as local
from datasets.sql import Gittables, Maintables, OpenData
import datasets.sql.csv_cache as csv_cache
import algorithms.naive_algorithm as naive_algorithm
import algorithms.machine_learning as machine_learning
import testing

load_dotenv()
db_params = {
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
}

scoring_strategies = [
    [['recall', 'precision'], [recall, precision]],
    # [['recall', 'recall', 'precision'], [recall, recall, precision]]
]


def main():
    # TODO: Key error: 1
    testing.prepare_and_train(row_count_iter=[5],
                              train_table_count=100,
                              data_path='src/data/gittables',
                              train_envenly=False,
                              scoring_strategies=scoring_strategies)

def random_int():
    testing.test_random_int(row_counts=[100, 1000, 10000, 100000, 1000000, 5000000, 10000000],
                            ncols=100,
                            out_path="test-random-int.csv",
                            path_to_model='src/data/model/10_rows/10000_tables/gittables/180minutes/recall_precision.pickle',
                            nrows=10)


if __name__ == '__main__':
    main()
