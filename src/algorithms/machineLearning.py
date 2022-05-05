import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
from typing import Iterable, Iterator
from pathlib import Path

from autosklearn.classification import AutoSklearnClassifier
from sklearn.model_selection import train_test_split
import pickle

import datasets.sql.csv_cache as csv_cache
import algorithms.naiveAlgorithm as naiveAlgorithm
import algorithms.machineLearning as machineLearning


header = ["Duplicates", "Data Type", "Sorted",
          # number
          "Min. value", "Max. value", "Mean", "Std. Deviation",
          # string
          "Avg. string length", "Min. string length", "Max. string length"
          # date?
          ]  # 10


def find_unique_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Generate a list with all column ids which only contain unique values making use of machine learning.

    Args:
        table (pd.Dataframe): the table to inspect

    Returns:
        pd.DataFrame: the indexes of the unique columns
    """
    print(prepare_table(table))


def prepare_table(table: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(columns=header)
    for column_id in table:
        result = pd.concat([result, prepare_column(
            table[column_id])])
    return result


def prepare_column(column: pd.DataFrame) -> pd.DataFrame:
    # return immediatly if there are any duplicated
    if column.duplicated().any():
        # 10
        return pd.DataFrame([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], columns=header, index=[column.name])
    # duplicate = 0, data type and sorted will be changed
    result = [0, 0, 0]
    # check if entries are sorted
    try:
        if all(column[i+1] >= column[i] for i in range(0, len(column)-1)):
            result[2] = 1
        if all(column[i+1] <= column[i] for i in range(0, len(column)-1)):
            result[2] = 1
    except:
        print(f"Column {column.name} does not just include strings")
    # handle integer and float
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
        except:
            result[1] = 3  # mixed column
            result += [0, 0, 0]
        return pd.DataFrame([result], columns=header, index=[column.name])
    raise NotImplementedError("Not implemented column type")


def _describe_string(column: pd.DataFrame) -> list:
    # "Avg. string length", "Min. string length", "Max. string length"
    length_list = []
    for value in column.values:
        if isinstance(value, str):
            length_list.append(len(value))
        else:
            raise ValueError("Not a String")
    average = sum(length_list)/len(length_list)
    minimum = min(length_list)
    maximum = max(length_list)
    return [average, minimum, maximum]


def prepare_training(table_range: Iterable, number_rows: int, non_trivial: bool, csv_path: str, path='src/data/training/'):
    if non_trivial:
        path = f"{path}{min(table_range)}-{max(table_range)}_{number_rows}_nt.csv"
    else:
        path = f"{path}{min(table_range)}-{max(table_range)}_{number_rows}.csv"
    path_result = path.replace(".csv", "-result.csv")
    pd.DataFrame([], columns=machineLearning.header).to_csv(path, index=False)
    pd.DataFrame([], columns=["PK Candidates"]).to_csv(
        path_result, index=False)
    for tableid in table_range:
        # TODO: error catching etc.
        table = csv_cache.get_table_local(csv_path, tableid, number_rows)
        data = machineLearning.prepare_table(table)
        if non_trivial:
            # remove all trivial cases
            trivial_cases = data[data["Duplicates"] == 1].index
            data = data.drop(trivial_cases)
        data.to_csv(path, mode='a', header=False, index=False)
        data = naiveAlgorithm.find_unique_columns(table)
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


def train(train_csv: str, save_path: str = "", train_time=120, per_run_time=30) -> AutoSklearnClassifier:
    X = pd.read_csv(train_csv)
    y = pd.read_csv(train_csv.replace('.csv', '-result.csv'))

    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    print("Starting Training")

    automl = AutoSklearnClassifier(
        time_left_for_this_task=train_time,
        per_run_time_limit=per_run_time,
    )
    # automl.fit(X_train, y_train, dataset_name="Test")
    automl.fit(X, y, dataset_name="Test")

    if save_path != "":
        Path(save_path.rsplit('/', 1)[0]).mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as file:
            pickle.dump(automl, file)
    return automl


def prepare_training_iterator(table_iter: Iterator, non_trivial: bool, read_tables_max: int, out_path='src/training/'):
    if non_trivial:
        out_path = f"{out_path}training-nontrivial.csv"
    else:
        out_path = f"{out_path}training.csv"
    path_result = out_path.replace(".csv", "-result.csv")
    pd.DataFrame([], columns=machineLearning.header).to_csv(
        out_path, index=False)
    pd.DataFrame([], columns=["PK Candidates"]).to_csv(
        path_result, index=False)
    count = 0
    for table in table_iter:
        count += 1
        if read_tables_max > 0 and count > read_tables_max:
            break
        data = machineLearning.prepare_table(table)
        if non_trivial:
            # remove all trivial cases
            trivial_cases = data[data["Duplicates"] == 1].index
            data = data.drop(trivial_cases)
        data.to_csv(out_path, mode='a', header=False, index=False)
        data = naiveAlgorithm.find_unique_columns_in_table(table)
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
