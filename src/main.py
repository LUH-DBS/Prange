import os
from typing import Iterable
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


def main():
    pass


def testcase_1(nrows_iter: Iterable[int], train_model: bool = False):
    """Train and test models which look at nrows rows for their prediction.

    Args:
        nrows_iter (Iterable[int]): A model will be trained/tested for each item in the Iterable.
        train_model (bool, optional): Only train the models if True. Defaults to False.
    """
    scoring_strategies = [
        [['recall', 'precision'], [recall, precision]],
        # [['recall', 'recall', 'precision'], [recall, recall, precision]]
    ]
    ntables = 5000
    if train_model:
        testing.prepare_and_train(row_count_iter=[5, 10, 20],
                                  train_table_count=10000,
                                  data_path='src/data/gittables',
                                  train_envenly=False,
                                  scoring_strategies=scoring_strategies)
    for nrows in nrows_iter:
        testing.test_model(path_to_model=f'src/data/model/{nrows}_rows/10000_tables/gittables/180minutes/recall_precision.pickle',
                           nrows=nrows,
                           input_path='src/data/gittables/',
                           output_path=f'test-{ntables}_{nrows}rows.csv',
                           files_per_dir=ntables,
                           skip_tables=-1)


def random_int():
    testing.test_random_int(row_counts=[100, 1000, 10000, 100000, 1000000, 5000000, 10000000],
                            ncols=100,
                            out_path="test-random-int.csv",
                            path_to_model='src/data/model/10_rows/10000_tables/gittables/180minutes/recall_precision.pickle',
                            nrows=10)


if __name__ == '__main__':
    main()
