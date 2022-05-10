"""A module containing functions to test the ml model."""

import os
import pickle
from typing import Iterable, Iterator
from genericpath import exists
import csv
from timeit import default_timer as timer

from autosklearn.classification import AutoSklearnClassifier

from datasets import local
from algorithms import naive_algorithm
from algorithms import machine_learning

BASE_PATH_TRAINING = 'src/data/training/'
BASE_PATH_MODEL = 'src/data/model/'


def test_model(path_to_model: str, nrows: int, input_path: str, output_path: str, skip_tables: int = -1) -> None:
    """Test a model and print the results into a csv file.

    Args:
        path_to_model (str): The filepath to the pickle file of the model.
        nrows (int): The number of rows the model will be inspecting.
        input_path (str): The path to the directory where the tables are located.
        output_path (str): The filepath where the result csv will be saved.
        skip_tables (int, optional): Skip the first [skip_tables] tables. Defaults to -1.
    """
    with open(path_to_model, 'rb') as file:
        ml = pickle.load(file)
    with open(output_path, 'w') as file:
        csv_file = csv.writer(file)
        row = ["Table Name", "Rows", "Columns", "Accuracy", "Precision",
               "Recall", "F1", "Time ML (usec)", "Time Naive (usec)"]
        csv_file.writerow(row)
        for table_path in local.traverse_directory_path(input_path, skip_tables=skip_tables, files_per_dir=5):
            ml_time = -timer()
            table = local.get_table(table_path, nrows)
            ml_unqiues = machine_learning.find_unique_columns(
                table.head(nrows), ml)
            ml_time += timer()
            na_time = -timer()
            table = local.get_table(table_path)
            naive_uniques = naive_algorithm.find_unique_columns_in_table(table)
            na_time += timer()
            true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
            for i in range(0, len(table.columns)):
                if i in ml_unqiues:
                    if i in naive_uniques:
                        true_pos += 1
                    else:
                        false_pos += 1
                else:
                    if i in naive_uniques:
                        false_neg += 1
                    else:
                        true_neg += 1
            accuracy = (true_pos + true_neg) / \
                (true_pos + true_neg + false_pos + false_neg)
            if true_pos + false_pos != 0:
                precision = true_pos / (true_pos + false_pos)
            else:
                precision = 0.0
            if true_pos + false_neg != 0:
                recall = true_pos / (true_pos + false_neg)
            else:
                recall = 1.0
            f1 = 2 * precision * recall / (precision + recall)
            row = [table_path.rsplit('/', 1)[1], *table.shape, accuracy, precision,
                   recall, f1, ml_time, na_time]
            csv_file.writerow(row)


def prepare_by_rows(row_count_iter: Iterable[int], train_table_count: int, data_path: str, files_per_dir: int = -1):
    """Prepare and conduct the training ml models. One model will be trained for each row_count in [row_count_iter].

    Args:
        row_count_iter (Iterable[int]): An Iterable containing different row_counts with witch a model will be trained.
        train_table_count (int): The number of tables to use to train the model.
        data_path (str): The path where the model will be saved as a pickle file.
    """
    for row_count in row_count_iter:
        training_csv_path = f'{BASE_PATH_TRAINING}{row_count}_rows/{train_table_count}_tables/{data_path.replace("src/data/", "")}/'
        table_iter = local.traverse_directory(
            data_path, row_count, files_per_dir)
        machine_learning.prepare_training_iterator(
            table_iter, False, train_table_count, training_csv_path)


def train_if_not_exists(train_csv: str, save_path: str, scoring_function_names: list[str], scoring_functions: list, train_time: int = 120, per_run_time: int = 30) -> AutoSklearnClassifier:
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
    save_path = f"{save_path}{train_time_minute}minutes/{'_'.join(scoring_function_names)}.pickle"
    if exists(save_path):
        with open(save_path, 'rb') as file:
            return pickle.load(file)
    else:
        return machine_learning.train(train_csv, scoring_functions, save_path, train_time, per_run_time)


def train_and_override(train_csv: str, save_path: str, scoring_function_names: list[str], scoring_functions: list, train_time: int = 120, per_run_time: int = 30) -> AutoSklearnClassifier:
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
    save_path = f"{save_path}{train_time_minute}minutes/{'_'.join(scoring_function_names)}.pickle"
    return machine_learning.train(train_csv, scoring_functions, save_path, train_time, per_run_time)


def list_models(path: str) -> Iterator[str]:
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == '.pickle':
                yield f"{root}/{file}"
