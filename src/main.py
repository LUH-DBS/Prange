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
    [['recall', 'recall', 'precision'], [recall, recall, precision]]
]


def main():
    # TODO: Key error: 1
    testing.prepare_and_train(row_count_iter=[5, 10, 15, 20],
                              train_table_count=6000,
                              data_path='src/data/gittables',
                              train_envenly=True,
                              scoring_strategies=scoring_strategies)

    # testing.test_model(path_to_model='src/data/model/10_rows/100_tables/gittables/0minutes.pickle',
    #                    nrows=10,
    #                    input_path='src/data/gittables/object_tables/',
    #                    output_path='test.csv',
    #                    skip_tables=-1)


if __name__ == '__main__':
    main()
