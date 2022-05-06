from genericpath import exists
import pickle
from typing import Iterable

from autosklearn.classification import AutoSklearnClassifier

import datasets.local as local
import algorithms.naiveAlgorithm as naiveAlgorithm
import algorithms.machineLearning as machineLearning


def prepare_and_train(row_count_iter: Iterable, data_path: str):
    for row_count in row_count_iter:
        table_iter = local.traverse_directory(data_path, row_count)
        # machineLearning.prepare_training_iterator(
        #     table_iter, False, 2000, 'src/data/training/')
        train_if_not_exists(row_count, 'src/data/training/training.csv',
                            f'src/data/model/{row_count}_rows/')


def train_if_not_exists(row_count: int, train_csv: str, save_path: str, train_time: int = 120, per_run_time: int = 30) -> AutoSklearnClassifier:
    train_time_minute = int(train_time / 60)
    save_path = f"{save_path}{train_time_minute}minutes.pickle"
    if exists(save_path):
        with open(save_path, 'rb') as file:
            return pickle.load(file)
    else:
        return machineLearning.train(train_csv, save_path, train_time, per_run_time)
