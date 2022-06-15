"""
This module holds all functions necessary to interact with the ml algorithm.
"""
from timeit import default_timer as timer
from typing import Iterable, Iterator, TextIO
from pathlib import Path
import pickle

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_bool_dtype
import numpy as np

# from autosklearn.classification import AutoSklearnClassifier
from autosklearn.experimental.askl2 import AutoSklearn2Classifier as AutoSklearnClassifier

from datasets.sql import csv_cache
from algorithms import naive_algorithm
import logging
logger = logging.getLogger(__name__)


header = ["Duplicates", "Data Type", "Sorted",
          # number
          "Min. value", "Max. value", "Mean", "Std. Deviation",
          # string
          "Avg. string length", "Min. string length", "Max. string length"
          # date?
          ]  # 10


def find_unique_columns(table: pd.DataFrame, model: AutoSklearnClassifier) -> list[int]:
    """Generate a list with all column ids which only contain unique values making use of machine learning.

    Args:
        table (pd.Dataframe): the table to inspect

    Returns:
        pd.DataFrame: the indexes of the unique columns
    """
    preparation_time = -timer()
    prepared_table = prepare_table(table)
    preparation_time += timer()
    prediction_time = -timer()
    prediction = model.predict(prepared_table)
    prediction_time += timer()
    logger.debug(
        f"Preparation: {preparation_time}s, prediction: {prediction_time}s")
    return [i for i in range(0, len(prediction)) if prediction[i] == 1]


def prepare_table(table: pd.DataFrame) -> pd.DataFrame:
    """Returns a table where each row contains the features of one column in the table.

    Args:
        table (pd.DataFrame): the table to inspect

    Returns:
        pd.DataFrame: the feature table
    """
    result = pd.DataFrame(columns=header)
    for column_id in table:
        result = pd.concat([result, prepare_column(
            table[column_id])])
    return result


def prepare_column(column: pd.Series) -> pd.DataFrame:
    """Extract the features of a single column.

    Args:
        column (pd.DataFrame): the column to inspect

    Raises:
        NotImplementedError: If a column dtype is encountered that is not accounted for.

    Returns:
        pd.DataFrame: the features of the column as a row
    """
    # return immediatly if there are any duplicated
    if column.duplicated().any():
        # 10
        return pd.DataFrame([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], columns=header, index=[column.name])
    none_value = False
    for value in column.values:
        if value == None or value == np.NaN:
            if none_value:
                return pd.DataFrame([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], columns=header, index=[column.name])
            else:
                none_value = True

    # use this code to recognize tables containing a single NaN as sorted (if the rest is sorted)
    # if none_value:
    #     column = column.copy().dropna()

    # duplicate = 0, data type and sorted will be changed
    result = [0, 0, 0]
    # check if entries are sorted
    try:
        if all(column[i+1] >= column[i] for i in range(0, len(column)-1)):
            result[2] = 1
        if all(column[i+1] <= column[i] for i in range(0, len(column)-1)):
            result[2] = 1
    except TypeError:
        logger.common_error(
            f"Column {column.name} does not just include strings (TypeError)")
    except KeyError as error:
        logger.common_error(
            f"KeyError with column {column.name}")
    # handle integer and float
    if is_bool_dtype(column):
        result[1] = 3
        result += [0, 0, 0, 0, 0, 0, 0]
        return pd.DataFrame([result], columns=header, index=[column.name])
    if is_numeric_dtype(column):
        result[1] = 1
        description = column.describe()
        result.append(description['min'])
        result.append(description['max'])
        result.append(description['mean'])
        result.append(description['std'])
        # values for strings
        result += [0, 0, 0]
        return pd.DataFrame([result], columns=header, index=[column.name])
    if is_string_dtype(column):
        result[1] = 2
        # values for numbers
        result += [0, 0, 0, 0]
        try:
            result += _describe_string(column)
        except ValueError:
            result[1] = 4  # mixed column
            result += [0, 0, 0]
        return pd.DataFrame([result], columns=header, index=[column.name])
    raise NotImplementedError("Not implemented column type")


def _describe_string(column: pd.DataFrame) -> list:
    """Return the features of a string column.

    Args:
        column (pd.DataFrame): the column of dtype string to inspect

    Raises:
        ValueError: if the column does not only include strings (possible with dtype string)

    Returns:
        list: a list with the features for strings
    """
    # "Avg. string length", "Min. string length", "Max. string length"
    length_list = []
    for value in column.values:
        if isinstance(value, str):
            length_list.append(len(value))
        else:
            raise ValueError("Not a String")
    if len(length_list) == 0:
        average = 0
    else:
        average = sum(length_list)/len(length_list)
    minimum = min(length_list)
    maximum = max(length_list)
    return [average, minimum, maximum]


def prepare_training(table_range: Iterable, number_rows: int, non_trivial: bool, csv_path: str, path='src/data/training/'):
    """Prepare a feature table to train a model on from the csv cache files.

    Args:
        table_range (Iterable): a range of table ids to use
        number_rows (int): the number of rows that will be read from each table
        non_trivial (bool): if True the columns which have duplicates in the first [number_rows] rows won't be saved
        csv_path (str): the path where the csv files are saved
        path (str, optional): the output path. Defaults to 'src/data/training/'.
    """
    if non_trivial:
        path = f"{path}{min(table_range)}-{max(table_range)}_{number_rows}_nt.csv"
    else:
        path = f"{path}{min(table_range)}-{max(table_range)}_{number_rows}.csv"
    path_result = path.replace(".csv", "-result.csv")
    pd.DataFrame([], columns=header).to_csv(path, index=False)
    pd.DataFrame([], columns=["PK Candidates"]).to_csv(
        path_result, index=False)
    for tableid in table_range:
        # TODO: error catching etc.
        table = csv_cache.get_table_local(csv_path, tableid, number_rows)
        data = prepare_table(table)
        if non_trivial:
            # remove all trivial cases
            trivial_cases = data[data["Duplicates"] == 1].index
            data = data.drop(trivial_cases)
        data.to_csv(path, mode='a', header=False, index=False)
        data = naive_algorithm.find_unique_columns_in_table_with_panda(table)
        filtered_data = []
        for i in range(0, len(table.columns)):
            if i in data:
                filtered_data.append(True)
            else:
                filtered_data.append(False)
        index = table.columns.values
        filtered_data = [int(x) for x in filtered_data]
        result = pd.DataFrame(filtered_data, index=index,
                              columns=["PK Candidate"])
        if non_trivial:
            result = result.drop(trivial_cases)
        result.to_csv(path_result, mode='a', header=False, index=False)


def train(train_csv: str, scoring_functions: list, save_path: str = "", train_time=120) -> AutoSklearnClassifier:
    """Train a network on a feature table.

    Args:
        train_csv (str): the path to the feature table
        save_path (str, optional): the path to save the model to (as a pickle file). Defaults to "".
        train_time (int, optional): number of seconds to train the network. Defaults to 120.

    Returns:
        AutoSklearnClassifier: the trained model
    """
    X = pd.read_csv(train_csv)
    y = pd.read_csv(train_csv.replace('.csv', '-result.csv'))

    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    logger.info("Starting Training for %d seconds", train_time)

    automl = AutoSklearnClassifier(
        time_left_for_this_task=train_time,
        scoring_functions=scoring_functions,
        memory_limit=200000,  # 200GB
    )
    # automl.fit(X_train, y_train, dataset_name="Test")
    automl.fit(X, y, dataset_name="Test")
    logger.info("Finished training")

    if save_path != "":
        Path(save_path.rsplit('/', 1)[0]).mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as file:
            pickle.dump(automl, file)
    return automl


def prepare_training_iterator(table_iter: Iterator[pd.DataFrame], non_trivial: bool, read_tables_max: int, out_path='src/training/'):
    """Prepare a feature table for training a model.

    Args:
        table_iter (Iterator): An Iterator that iterates over the DataFrames from the .parquet files to use for the training.
        non_trivial (bool): If True, columns which have duplicates in the first n rows will be skipped.
        read_tables_max (int): return after this many tables
        out_path (str, optional): the path where to save the feature table. Defaults to 'src/training/'.
    """
    if non_trivial:
        out_path = f"{out_path}training-nontrivial.csv"
    else:
        out_path = f"{out_path}training.csv"
    Path(out_path.rsplit('/', 1)[0]).mkdir(parents=True, exist_ok=True)
    path_result = out_path.replace(".csv", "-result.csv")
    pd.DataFrame([], columns=header).to_csv(
        out_path, index=False)
    pd.DataFrame([], columns=["PK Candidates"]).to_csv(
        path_result, index=False)
    count = 0
    for table in table_iter:
        count += 1
        if read_tables_max > 0 and count > read_tables_max:
            break
        data = prepare_table(table)
        if non_trivial:
            # remove all trivial cases
            trivial_cases = data[data["Duplicates"] == 1].index
            data = data.drop(trivial_cases)
        data.to_csv(out_path, mode='a', header=False, index=False)
        data = naive_algorithm.find_unique_columns_in_table_with_panda(table)
        filtered_data = []
        for i in range(0, len(table.columns)):
            if i in data:
                filtered_data.append(True)
            else:
                filtered_data.append(False)
        index = table.columns.values
        filtered_data = [int(x) for x in filtered_data]
        result = pd.DataFrame(filtered_data, index=index,
                              columns=["PK Candidate"])
        if non_trivial:
            result = result.drop(trivial_cases)
        result.to_csv(path_result, mode='a', header=False, index=False)
