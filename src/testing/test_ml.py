"""A module containing functions to test the ml model."""

import os
import pickle
import pandas as pd
import numpy as np
from pyarrow.lib import ArrowInvalid
from typing import Iterable, Iterator
from pathlib import Path
from shutil import rmtree
from genericpath import exists
import csv
from timeit import default_timer as timer

from autosklearn.classification import AutoSklearnClassifier

from datasets import local
from algorithms import naive_algorithm
from algorithms import machine_learning
import logging
logger = logging.getLogger(__name__)

BASE_PATH_TRAINING = 'src/data/training/'
BASE_PATH_MODEL = 'src/data/model/'


def test_model(path_to_model: str, nrows: int, input_path: str, output_path: str, files_per_dir: int, use_small_tables: bool, skip_tables: int = -1) -> None:
    """Test a model and print the results into a csv file.

    Args:
        path_to_model (str): The filepath to the pickle file of the model.
        nrows (int): The number of rows the model will be inspecting.
        input_path (str): The path to the directory where the tables are located.
        output_path (str): The filepath where the result csv will be saved.
        use_small_tables (bool): If True, only the first nrows will be loaded for the model and only positiv columns for the validation.
        skip_tables (int, optional): Skip the first [skip_tables] tables. Defaults to -1.
    """
    logger.info("Started testing of a model with %s rows", nrows)
    with open(path_to_model, 'rb') as file:
        ml = pickle.load(file)
    table_path_list: list[str] = []
    ml_dict = {}
    naive_dict = {}
    counter = 0
    if 'csv' in input_path:
        use_small_tables = False
    elif use_small_tables:
        output_path = output_path.replace('.csv', '_small-tables.csv')
    with open(output_path, 'w') as file:
        csv_file = csv.writer(file)
        row = ["Table Name", "Rows", "Columns", "Accuracy", "Precision",
               "Recall", "F1", "ML: Loading", "ML: Compute Time", "ML: Loading", "ML: Validation Time", "ML: Total", "Naive: Loading", "Naive: Compute Time", "Naive: Total", "True Pos", "True Neg", "False Pos", "False Neg"]
        csv_file.writerow(row)
    for table_path in local.traverse_directory_path(input_path, skip_tables=skip_tables, files_per_dir=files_per_dir):
        if counter % 100 == 0 and counter != 0:
            logger.info("Finished model testing of %s tables", counter)
        counter += 1
        logger.debug(
            f"Model on table {counter} ({table_path.rsplit('/', 1)[1]})")
        try:
            total_time = -timer()
            load_time = -timer()
            if use_small_tables:
                small_table = local.get_table(table_path, nrows)
            else:
                table = local.get_table(table_path)
                small_table = table.head(nrows)
            load_time += timer()
            computing_time = -timer()
            unique_columns = machine_learning.find_unique_columns(
                small_table, ml)
            computing_time += timer()
            load_time2 = -timer()
            if use_small_tables:
                table = local.get_table(
                    table_path, columns=small_table.columns[unique_columns])
            load_time2 += timer()
            # TODO: validate columns
            confirmed_time = -timer()
            confirmed_time += timer()
            total_time += timer()
            ml_dict[table_path] = {
                'unique_columns': unique_columns,
                'load_time': load_time,
                'computing_time': computing_time,
                'load_time2': load_time2,
                'confirmed_time': confirmed_time,
                'total_time': total_time
            }
            table_path_list.append(table_path)
        except pd.errors.ParserError as e:
            counter -= 1
            logger.common_error(
                "ParserError with file %s", table_path)
            continue
        except ArrowInvalid as error:
            counter -= 1
            logger.common_error(
                "ArrowInvalid error with file %s", table_path)
            continue
        except UnicodeDecodeError:
            counter -= 1
            logger.common_error(
                "UnicodeDecodeError with file %s", table_path)
            continue
    with open(output_path, 'a') as file:
        csv_file = csv.writer(file)
        for table_path in table_path_list:
            try:
                total_time = -timer()
                load_time = -timer()
                table = local.get_table(table_path)
                load_time += timer()
                computing_time = -timer()
                unique_columns = naive_algorithm.find_unique_columns_in_table(
                    table)
                computing_time += timer()
                total_time += timer()
                naive_dict[table_path] = {
                    'unique_columns': unique_columns,
                    'load_time': load_time,
                    'computing_time': computing_time,
                    'total_time': total_time
                }
            except pd.errors.ParserError as e:
                counter -= 1
                logger.common_error(
                    "ParserError with file %s", table_path)
                continue
            except ArrowInvalid as error:
                counter -= 1
                logger.common_error(
                    "ArrowInvalid error with file %s", table_path)
                continue
            except UnicodeDecodeError:
                counter -= 1
                logger.common_error(
                    "UnicodeDecodeError with file %s", table_path)
                continue

            ml_values = ml_dict[table_path]
            naive_values = naive_dict[table_path]
            true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
            for i in range(0, len(table.columns)):
                if i in ml_values['unique_columns']:
                    if i in naive_values['unique_columns']:
                        true_pos += 1
                    else:
                        false_pos += 1
                else:
                    if i in naive_values['unique_columns']:
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
                   recall, f1, ml_values['load_time'], ml_values['computing_time'], ml_values['load_time2'], ml_values['confirmed_time'], ml_values['total_time'], naive_values['load_time'], naive_values['computing_time'], naive_values['total_time'], true_pos, true_neg, false_pos, false_neg]
            csv_file.writerow(row)
    logger.info("Finished testing")


def prepare_by_rows(row_count_iter: Iterable[int], train_table_count: int, data_path: str, files_per_dir: int = -1):
    """Prepare the training data. One trainingsset will be generated for each item in [row_count_iter].

    Args:
        row_count_iter (Iterable[int]): An Iterable containing different row_counts for which the trainingsset will be generated.
        train_table_count (int): The number of tables to use to train the model.
        data_path (str): The path where the tables used for training are.
        files_per_dir (int): If > 0, only this many files in each subdirectory will be used. Defaults to -1. 
    """
    for row_count in row_count_iter:
        logger.info("Started preparing trainingsdata from %s tables using %s rows",
                    train_table_count, row_count)
        training_csv_path = f'{BASE_PATH_TRAINING}{row_count}_rows/{train_table_count}_tables/{data_path.replace("src/data/", "")}/'
        table_iter = local.traverse_directory(
            data_path, row_count, files_per_dir)
        machine_learning.prepare_training_iterator(
            table_iter, False, train_table_count, training_csv_path)
        logger.info("Finished preparation")


def prepare_and_train(row_count_iter: Iterable[int], train_table_count: int, data_path: str, train_envenly: bool, scoring_strategies: list[list[list]], train_time: int):
    """Prepare and execute the training of one ml model per value of [row_count_iter] and per value of [scoring_strategies].

    Args:
        row_count_iter (Iterable[int]): Number of rows to train the model on.
        train_table_count (int): The number of tables to use for the training.
        data_path (str): The path to the directory where the table files are.
        train_envenly (bool): If True, the tables will be evenly from the subdirectories of [data_path].
        train_time (int): Number of seconds to train the model.
    """
    # per_run_time = 300  # 5 minutes
    per_run_time = int(train_time / 10)
    files_per_dir = -1
    number_of_subdirs = len(
        [f.path for f in os.scandir(data_path) if f.is_dir()])
    if train_envenly and number_of_subdirs > 0:
        files_per_dir = int(train_table_count / number_of_subdirs)
    prepare_by_rows(row_count_iter=row_count_iter,
                    train_table_count=train_table_count,
                    data_path=data_path,
                    files_per_dir=files_per_dir)
    for row_count in row_count_iter:
        for strategy in scoring_strategies:
            training_csv_path = f'src/data/training/{row_count}_rows/{train_table_count}_tables/{data_path.replace("src/data/", "")}/'
            model_path = f'src/data/model/{row_count}_rows/{train_table_count}_tables/{data_path.replace("src/data/", "")}/'
            train_model(train_csv=training_csv_path + 'training.csv',
                        scoring_function_names=strategy[0],
                        scoring_functions=strategy[1],
                        save_path=model_path,
                        train_time=train_time,
                        per_run_time=per_run_time,
                        train_if_exists=True
                        )


def train_model(train_csv: str, save_path: str, scoring_function_names: list[str], scoring_functions: list, train_time: int = 120, per_run_time: int = 30, train_if_exists: bool = False) -> AutoSklearnClassifier:
    """Train and save a ml model if it doesn't already exist.

    Args:
        train_csv (str): path to the feature table used to train the model
        save_path (str): directory where to save the model (ends with /)
        scoring_function_names (list[str]): A list with the names of the scoring functions for the filenames.
        scoring_functions (list): A list with the scoring functions for the training.
        train_time (int, optional): number of seconds to train the network. Defaults to 120.
        per_run_time (int, optional): number of seconds for each run. Defaults to 30.
        train_if_exists (bool, optional): If True, a new model will be trained even if one with the given parameters already exists. Defaults to False.

    Returns:
        AutoSklearnClassifier: The trained model.
    """
    train_time_minute = int(train_time / 60)
    save_path = f"{save_path}{train_time_minute}minutes/{'_'.join(scoring_function_names)}.pickle"
    if exists(save_path) and not train_if_exists:
        logger.info("Loading model from %s", save_path)
        with open(save_path, 'rb') as file:
            return pickle.load(file)
    else:
        logger.info("Started to train a model for %s minutes (%s)",
                    train_time_minute, ", ".join(scoring_function_names))
        model = machine_learning.train(train_csv=train_csv,
                                       scoring_functions=scoring_functions,
                                       save_path=save_path,
                                       train_time=train_time,
                                       per_run_time=per_run_time
                                       )
        logger.info("Finished training")
        return model


def list_models(path: str) -> Iterator[str]:
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == '.pickle':
                yield f"{root}/{file}"


def generate_random_int_dataframe(nrows: int, ncols: int) -> pd.DataFrame:
    logger.info(
        f"Generating random_int table with {nrows:,d} rows and {ncols:,d} columns")
    min_number = 0
    max_number = nrows
    col_names = [f"Row {i}" for i in range(0, ncols)]
    return pd.DataFrame(np.random.randint(
        min_number, max_number, size=(nrows, ncols)), columns=col_names)


def test_random_int(row_counts: list[int], ncols: int, out_path: str, path_to_model: str, model_rows: int, nrows: int, use_small_tables: bool, csv: bool = False, generate_tables: bool = True) -> None:
    path = 'src/data/generated/'
    if exists(path) and generate_tables:
        rmtree(path)
    Path(path).mkdir(parents=True, exist_ok=True)
    if generate_tables:
        for nrows in row_counts:
            if csv:
                filepath = f'src/data/generated/{nrows}-{ncols}.csv'
                generate_random_int_dataframe(nrows, ncols).to_csv(filepath)
            else:
                filepath = f'src/data/generated/{nrows}-{ncols}.parquet'
                generate_random_int_dataframe(
                    nrows, ncols).to_parquet(filepath)
    test_model(path_to_model=path_to_model,
               nrows=model_rows,
               input_path=path,
               output_path=out_path,
               files_per_dir=100000,
               skip_tables=-1,
               use_small_tables=use_small_tables
               )
