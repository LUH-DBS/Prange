"""A module containing functions to test the ml model."""

import pickle
from typing import Iterable
from genericpath import exists

from autosklearn.classification import AutoSklearnClassifier

from datasets import local
from algorithms import naive_algorithm
from algorithms import machine_learning

BASE_PATH_TRAINING = 'src/data/training/'
BASE_PATH_MODEL = 'src/data/model/'


def prepare_and_train(row_count_iter: Iterable[int], train_table_count: int, data_path: str, scoring_functions: list):
    """Prepare and conduct the training ml models. One model will be trained for each row_count in [row_count_iter].

    Args:
        row_count_iter (Iterable[int]): An Iterable containing different row_counts with witch a model will be trained.
        train_table_count (int): The number of tables to use to train the model.
        data_path (str): The path where the model will be saved as a pickle file.
    """
    for row_count in row_count_iter:
        training_csv_path = f'{BASE_PATH_TRAINING}{row_count}_rows/{train_table_count}_tables/{data_path.replace("src/data/", "")}/'
        model_path = f'{BASE_PATH_MODEL}{row_count}_rows/{train_table_count}_tables/{data_path.replace("src/data/", "")}/'
        table_iter = local.traverse_directory(data_path, row_count)
        # training
        machine_learning.prepare_training_iterator(
            table_iter, False, train_table_count, training_csv_path)
        train_if_not_exists(train_csv=training_csv_path + "training.csv",
                            save_path=model_path,
                            train_time=30,
                            scoring_functions=scoring_functions)


def train_if_not_exists(train_csv: str, save_path: str, scoring_functions: list, train_time: int = 120, per_run_time: int = 30) -> AutoSklearnClassifier:
    """Train and save a ml model if it doesn't already exist.

    Args:
        train_csv (str): path to the feature table used to train the model
        save_path (str): directory where to save the model (ends with /)
        train_time (int, optional): number of seconds to train the network. Defaults to 120.
        per_run_time (int, optional): number of seconds for each run. Defaults to 30.

    Returns:
        AutoSklearnClassifier: _description_
    """
    train_time_minute = int(train_time / 60)
    save_path = f"{save_path}{train_time_minute}minutes.pickle"
    if exists(save_path):
        with open(save_path, 'rb') as file:
            return pickle.load(file)
    else:
        return machine_learning.train(train_csv, scoring_functions, save_path, train_time, per_run_time)
