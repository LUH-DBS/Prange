from genericpath import exists
import pickle
from typing import Iterable

from autosklearn.classification import AutoSklearnClassifier

import datasets.local as local
import algorithms.naiveAlgorithm as naiveAlgorithm
import algorithms.machineLearning as machineLearning

base_path_training = 'src/data/training/'
base_path_model = 'src/data/model/'


def prepare_and_train(row_count_iter: Iterable, train_table_count: int, test_table_count: int, data_path: str):
    for row_count in row_count_iter:
        training_csv_path = f'{base_path_training}{row_count}_rows/{train_table_count}_tables/{data_path.replace("src/data/", "")}/'
        model_path = f'{base_path_model}{row_count}_rows/{train_table_count}_tables/{data_path.replace("src/data/", "")}/'
        table_iter = local.traverse_directory(data_path, row_count)
        # training
        machineLearning.prepare_training_iterator(
            table_iter, False, train_table_count, training_csv_path)
        # testing
        machineLearning.prepare_training_iterator(
            table_iter, True, test_table_count, training_csv_path)
        train_if_not_exists(train_csv=training_csv_path + "training.csv",
                            save_path=model_path,
                            train_time=30)


def train_if_not_exists(train_csv: str, save_path: str, train_time: int = 120, per_run_time: int = 30) -> AutoSklearnClassifier:
    train_time_minute = int(train_time / 60)
    save_path = f"{save_path}{train_time_minute}minutes.pickle"
    if exists(save_path):
        with open(save_path, 'rb') as file:
            return pickle.load(file)
    else:
        return machineLearning.train(train_csv, save_path, train_time, per_run_time)
