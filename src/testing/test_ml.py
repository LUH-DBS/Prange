"""A module containing functions to test the ml model."""

import os
import pickle
import pandas as pd
import numpy as np
from pyarrow.lib import ArrowInvalid
from typing import Iterable, Iterator, List, Tuple
from pathlib import Path
from shutil import rmtree
from genericpath import exists
import csv
from timeit import default_timer as timer
from datetime import datetime
import re

from autosklearn.classification import AutoSklearnClassifier

from datasets import local
from algorithms import naive_algorithm
from algorithms import machine_learning
import logging
logger = logging.getLogger(__name__)

BASE_PATH_TRAINING = 'src/data/training/'
BASE_PATH_MODEL = 'src/data/model/'


def test_model(path_to_model: str, model_rows: int, input_path: str, output_path: str, use_small_tables: bool, speed_test: bool, max_files: int = -1, files_per_dir: int = -1, skip_tables: int = -1, min_rows: int = -1, min_cols: int = -1, log_false_guesses: bool = False) -> None:
    """Test a model and print the results into a csv file.

    Args:
        path_to_model (str): The filepath to the pickle file of the model.
        nrows (int): The number of rows the model will be inspecting.
        input_path (str): The path to the directory where the tables are located.
        output_path (str): The filepath where the result csv will be saved.
        use_small_tables (bool): If True, only the first nrows will be loaded for the model and only positiv columns for the validation.
        max_files (int, optional): The maximum files/tables that will be used. Defaults to -1.
        files_per_dir (int, optional): Use only this many files for each subdirectory of the datasource. Defaults to -1.
        skip_tables (int, optional): Skip the first `skip_tables` tables. Defaults to -1.
        min_rows (int, optional): Skip a table if it has less than `min_rows` rows. Defaults to -1.
        min_cols (int, optional): Skip a table if it has less than `min_cols` columns. Defaults to -1.

    """
    with open(path_to_model, 'rb') as file:
        ml = pickle.load(file)
    table_path_list: list[str] = []
    ml_dict = {}
    naive_dict = {}
    counter = 0
    skipcounter = 0
    abortcounter = 0
    TIME_STRING = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    if 'csv' in input_path:
        use_small_tables = False
    elif use_small_tables:
        output_path = output_path.replace('.csv', '_small-tables.csv')
    with open(output_path, 'w') as file:
        csv_file = csv.writer(file)
        if speed_test:
            row = ["Table Name", "Rows", "Columns", "ML: Loading I", "ML: Compute Time", "ML: Loading II",
                   "ML: Validation Time", "ML: Total", "Naive: Loading", "Naive: Compute Time", "Naive: Total", "True Pos", "True Neg", "False Pos", "False Neg"]
        else:
            row = ["Table Name", "Rows", "Columns", "Accuracy", "Precision",
                   "Recall", "F1", "ML: Compute Time", "ML: Validation Time", "ML: Total", "Naive: Compute Time", "Naive: Total", "True Pos", "True Neg", "False Pos", "False Neg"]
        csv_file.writerow(row)
    for table_path in local.traverse_directory_path(input_path, files_per_dir=files_per_dir):
        try:
            table = local.get_table(table_path)
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
        # skip this table if it is smaller than necessary
        if len(table.columns) < min_cols:
            # logger.debug("Table to small (columns), aborting")
            abortcounter += 1
            continue
        # skip this table if it is smaller than necessary
        if len(table) < min_rows:
            # logger.debug("Table to small (rows), aborting")
            abortcounter += 1
            continue
        if skipcounter < skip_tables:
            skipcounter += 1
            continue
        if max_files > -1 and counter >= max_files:
            break
        counter += 1
        table_path_list.append(table_path)
    logger.info(
        f"Aborted {abortcounter} too small tables and skipped {skipcounter} tables to get to {counter} tables.")
    counter = 0
    logger.info("Started testing of a model with %s rows", model_rows)
    for table_path in table_path_list:
        if max_files > 0 and counter >= max_files:
            break
        if counter % 100 == 0 and counter != 0:
            logger.info("Finished model testing of %s tables", counter)
        counter += 1
        logger.debug(
            f"Model on table {counter} ({table_path.rsplit('/', 1)[1]})")
        try:
            total_time = -timer()
            # load the dataset
            load_time = -timer()
            if use_small_tables:
                # only get the first rows of the table
                small_table = local.get_table(table_path, model_rows)
            else:
                # get the whole table
                table = local.get_table(table_path)
                # use only the first rows for the model
                small_table = table.head(model_rows)
            load_time += timer()
            # use the model
            computing_time = -timer()
            # use the model to guess the unique columns
            unique_columns = machine_learning.find_unique_columns(
                small_table, ml)
            computing_time += timer()
            # load the rest of the table if not done in loading 1
            load_time2 = -timer()
            if use_small_tables:
                # get the columns which the model says are unique
                table = local.get_table(
                    table_path, columns=small_table.columns[unique_columns])
            else:
                # use only the columns which are unique according to the model for validation
                table = table[table.columns[unique_columns]]
            load_time2 += timer()
            # confirm the guess of the model
            confirmed_time = -timer()
            validated_uniques = naive_algorithm.find_unique_columns_in_table(
                table)
            confirmed_time += timer()
            total_time += timer()
            ml_dict[table_path] = {
                'unique_columns': unique_columns,
                'validated_uniques': validated_uniques,
                'load_time': load_time,
                'computing_time': computing_time,
                'load_time2': load_time2,
                'confirmed_time': confirmed_time,
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
    with open(output_path, 'a') as file:
        csv_file = csv.writer(file)
        counter = 0
        logger.info("Started testing of the naive algorithm")
        for table_path in table_path_list:
            counter += 1
            logger.debug(
                f"Naive algorithm on table {counter} ({table_path.rsplit('/', 1)[1]})")
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
                if log_false_guesses:
                    # log false negatives
                    false_neg = [
                        i for i in unique_columns if i not in ml_dict[table_path]['unique_columns']]
                    if len(false_neg) > 0:
                        logger.error(
                            "False Negativ (column '{}')".format("', '".join(table.columns[false_neg])))
                        log_path = f"src/result/correctness/false_neg/{TIME_STRING}/{model_rows}rows/"
                        Path(log_path).mkdir(parents=True, exist_ok=True)
                        false_neg_table = table[table.columns[false_neg]]
                        false_neg_table.to_csv(
                            log_path + table_path.rsplit('/', 1)[1] + '.csv')
                    # log false positives
                    false_pos = [
                        i for i in ml_dict[table_path]['unique_columns'] if i not in unique_columns]
                    if len(false_pos) > 0:
                        logger.debug(
                            "False Positiv (column '{}')".format("', '".join(table.columns[false_pos])))
                        log_path = f"src/result/correctness/false_pos/{TIME_STRING}/{model_rows}rows/"
                        Path(log_path).mkdir(parents=True, exist_ok=True)
                        false_pos_table = table[table.columns[false_pos]]
                        false_pos_table.to_csv(
                            log_path + table_path.rsplit('/', 1)[1] + '.csv')
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

            row = _make_row(ml_dict=ml_dict,
                            naive_dict=naive_dict,
                            speed=speed_test,
                            table=table,
                            table_path=table_path
                            )
            csv_file.writerow(row)
    logger.info("Finished testing")


def _make_row(speed: bool, ml_dict, naive_dict, table_path: str, table: pd.DataFrame):
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
    (accuracy, precision, recall, f1) = _compute_statistics(
        true_pos, true_neg, false_pos, false_neg, table_path)
    if speed:
        return [table_path.rsplit('/', 1)[1], *table.shape, ml_values['load_time'], ml_values['computing_time'], ml_values['load_time2'], ml_values['confirmed_time'], ml_values['total_time'], naive_values['load_time'], naive_values['computing_time'], naive_values['total_time'], true_pos, true_neg, false_pos, false_neg]
    else:
        return [table_path.rsplit('/', 1)[1], *table.shape, accuracy, precision,
                recall, f1, ml_values['computing_time'], ml_values['confirmed_time'], ml_values['total_time'], naive_values['computing_time'], naive_values['total_time'], true_pos, true_neg, false_pos, false_neg]
        # return [table_path.rsplit('/', 1)[1], *table.shape, accuracy, precision,
        #         recall, f1, ml_values['load_time'], ml_values['computing_time'], ml_values['load_time2'], ml_values['confirmed_time'], ml_values['total_time'], naive_values['load_time'], naive_values['computing_time'], naive_values['total_time'], true_pos, true_neg, false_pos, false_neg]


def _compute_statistics(true_pos: int, true_neg: int, false_pos: int, false_neg: int, table_path: str = "") -> Tuple[float, float, float, float]:
    try:
        accuracy = (true_pos + true_neg) / \
            (true_pos + true_neg + false_pos + false_neg)
    except ZeroDivisionError:
        logger.error(f"ZeroDivisionError with file {table_path} (accuracy)")
        accuracy = -1
    try:
        if true_pos + false_pos != 0:
            precision = true_pos / (true_pos + false_pos)
        else:
            precision = 0.0
    except ZeroDivisionError:
        logger.error(f"ZeroDivisionError with file {table_path} (precision)")
        precision = -1
    try:
        if true_pos + false_neg != 0:
            recall = true_pos / (true_pos + false_neg)
        else:
            recall = 1.0
    except ZeroDivisionError:
        logger.error(f"ZeroDivisionError with file {table_path} (recall)")
        recall = -1
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        logger.error(f"ZeroDivisionError with file {table_path} (f1)")
        f1 = -1
    return (accuracy, precision, recall, f1)


def prepare_by_rows(row_count_iter: Iterable[int], train_table_count: int, data_path: str, min_rows: int, min_cols: int, files_per_dir: int = -1):
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
        table_path_iter = local.traverse_directory_path(
            data_path, row_count, files_per_dir)
        machine_learning.prepare_training_iterator(table_path_iter=table_path_iter,
                                                   non_trivial=False,
                                                   read_tables_max=train_table_count,
                                                   out_path=training_csv_path,
                                                   min_rows=min_rows,
                                                   min_cols=min_cols
                                                   )
        logger.info("Finished preparation")


def prepare_and_train(row_count_iter: Iterable[int], train_table_count: int, data_path: str, train_envenly: bool, scoring_strategies: list[list[list]], train_time_list: List[int], min_rows: int, min_cols: int):
    """Prepare and execute the training of one ml model per value of [row_count_iter] and per value of [scoring_strategies].

    Args:
        row_count_iter (Iterable[int]): Number of rows to train the model on.
        train_table_count (int): The number of tables to use for the training.
        data_path (str): The path to the directory where the table files are.
        train_envenly (bool): If True, the tables will be evenly from the subdirectories of [data_path].
        train_time (int): Number of seconds to train the model.
    """
    files_per_dir = -1
    number_of_subdirs = len(
        [f.path for f in os.scandir(data_path) if f.is_dir()])
    if train_envenly and number_of_subdirs > 0:
        files_per_dir = int(train_table_count / number_of_subdirs)
    prepare_by_rows(row_count_iter=row_count_iter,
                    train_table_count=train_table_count,
                    data_path=data_path,
                    files_per_dir=files_per_dir,
                    min_rows=min_rows,
                    min_cols=min_cols
                    )
    for row_count in row_count_iter:
        for strategy in scoring_strategies:
            for train_time in train_time_list:
                logger.info(
                    f"Training model with {row_count} rows with strategy {strategy} for {int(train_time / 60)}minutes")
                training_csv_path = f'src/data/training/{row_count}_rows/{train_table_count}_tables/{data_path.replace("src/data/", "")}/'
                model_path = f'src/data/model/{row_count}_rows/{train_table_count}_tables/{data_path.replace("src/data/", "")}/'
                train_model(train_csv=training_csv_path + 'training.csv',
                            scoring_function_names=strategy[0],
                            scoring_functions=strategy[1],
                            save_path=model_path,
                            train_time=train_time,
                            train_if_exists=True
                            )


def train_model(train_csv: str, save_path: str, scoring_function_names: list[str], scoring_functions: list, train_time: int = 120, train_if_exists: bool = False) -> AutoSklearnClassifier:
    """Train and save a ml model if it doesn't already exist.

    Args:
        train_csv (str): path to the feature table used to train the model
        save_path (str): directory where to save the model (ends with /)
        scoring_function_names (list[str]): A list with the names of the scoring functions for the filenames.
        scoring_functions (list): A list with the scoring functions for the training.
        train_time (int, optional): number of seconds to train the network. Defaults to 120.
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
                                       train_time=train_time
                                       )
        logger.info("Finished training")
        return model


def list_models(path: str) -> Iterator[str]:
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == '.pickle':
                yield f"{root}/{file}"


def generate_random_int_dataframe(nrows: int, ncols: int, nonunique_percent: int) -> pd.DataFrame:
    logger.debug(
        f"Generating random_int table with {nrows:,d} rows and {ncols:,d} columns ({nonunique_percent}% nuniques)")
    # the number of columns which contain duplicates
    number_nonuniques = int(ncols * nonunique_percent / 100)
    # the first 50 rows are in ascending order to help the model guess correctly
    unique_cols_first = pd.DataFrame([[i] * (ncols - number_nonuniques)
                                      for i in range(0, 50)], columns=[f"Column {i}" for i in range(0, ncols - number_nonuniques)])
    unique_cols_rest = pd.DataFrame([[i] * (ncols - number_nonuniques)
                                     for i in range(50, nrows)], columns=[f"Column {i}" for i in range(0, ncols - number_nonuniques)])
    # the remaining rows are mixed up
    unique_cols_rest = unique_cols_rest.sample(frac=1).reset_index(drop=True)
    unique_cols = pd.concat(
        [unique_cols_first, unique_cols_rest], ignore_index=True)
    # the same is happening for the nonunique columns; the first two rows are duplicates
    nonunique_cols_first = pd.DataFrame([[nrows] * number_nonuniques
                                        for i in range(0, 2)], columns=[f"Column {i}" for i in range(ncols - number_nonuniques, ncols)])
    nonunique_cols_rest = pd.DataFrame([[i] * number_nonuniques
                                        for i in range(0, nrows - 2)], columns=[f"Column {i}" for i in range(ncols - number_nonuniques, ncols)])
    nonunique_cols_rest = nonunique_cols_rest.sample(
        frac=1).reset_index(drop=True)
    nonunique_cols = pd.concat(
        [nonunique_cols_first, nonunique_cols_rest], ignore_index=True)
    return pd.DataFrame(pd.concat([unique_cols, nonunique_cols], axis=1, join='inner'))


def test_random_int(row_counts: list[int], ncols: int, out_path: str, path_to_model: str, model_rows: int, use_small_tables: bool, nonunique_percent: int, csv: bool = False, generate_tables: bool = True) -> None:
    path = 'src/data/generated/'
    if exists(path) and generate_tables:
        rmtree(path)
    Path(path).mkdir(parents=True, exist_ok=True)
    if generate_tables:
        for nrows in row_counts:
            if csv:
                filepath = f'src/data/generated/{nrows}-{ncols}.csv'
                generate_random_int_dataframe(
                    nrows, ncols, nonunique_percent).to_csv(filepath, index=False)
            else:
                filepath = f'src/data/generated/{nrows}-{ncols}.parquet'
                generate_random_int_dataframe(
                    nrows, ncols, nonunique_percent).to_parquet(filepath)
    test_model(path_to_model=path_to_model,
               model_rows=model_rows,
               input_path=path,
               output_path=out_path,
               files_per_dir=-1,
               skip_tables=-1,
               use_small_tables=use_small_tables,
               speed_test=True
               )


def correctness_summary(input_dir: str, output_file: str):
    Path(output_file.rsplit('/', 1)[0]).mkdir(parents=True, exist_ok=True)
    summary_columns = ['Avg. rows', 'Columns', 'Accuracy', 'Precision', 'Recall',
                       'F1']  # , 'ML Time', 'Naive Time', 'True Pos', 'True Neg', 'False Pos', 'False Neg'
    result = []
    training_time = False
    input_size = False
    for tablepath in local.traverse_directory_path(input_dir):
        table = local.get_table(tablepath)
        avg_rows = table[['Rows']].sum() / table[['Rows']].count()
        sum_cols = table[['Columns']].sum()
        sum_results = table[['True Pos', 'True Neg',
                             'False Pos', 'False Neg']].sum()
        stats = _compute_statistics(*sum_results.values)
        # times = table[['ML: Total', 'Naive: Total']].sum()
        row = [*avg_rows.values, *sum_cols.values, *stats]  # , *times.values
        # Training Time
        tmp = re.findall("\d+minutes", tablepath)
        tmp = [re.findall("\d+", x) for x in tmp]
        if tmp and tmp[0]:
            training_time = True
            row.insert(0, tmp[0][0])
        # Model Input Size
        tmp = re.findall("\d+rows", tablepath)
        tmp = [re.findall("\d+", x) for x in tmp]
        if tmp and tmp[0]:
            input_size = True
            row.insert(0, tmp[0][0])
        result.append(row)
    if training_time:
        summary_columns.insert(0, 'Training Time')
    if input_size:
        summary_columns.insert(0, 'Model Input Size')
    result = pd.DataFrame(result, columns=summary_columns).sort_values(
        summary_columns[0])
    result.to_csv(output_file, index=False)
