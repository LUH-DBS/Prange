from email import header
from genericpath import exists
import os
from pprint import pprint
from typing import Iterable
from dotenv import load_dotenv
from sqlalchemy import create_engine
import sys
from pprint import pprint
import pandas as pd
import numpy as np
import algorithms

from datasets import Gittables, Maintables, OpenData
import datasets.csv as csv_interface

from algorithms import NaiveAlgorithm, MachineLearning

load_dotenv()
db_params = {
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
}


def main():
    engine = create_engine(
        f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}/{db_params['database']}")

    with engine.connect() as connection:
        global tables
        tables = Gittables(connection)
        global algorithm
        algorithm = MachineLearning()
        global csv_path
        csv_path = f"src/data/{tables.pathname()}"
        unique_columns(range(100, 101), True, True, True)
        # save_csv(range(200, 2200)) # opendata
        # csv_interface.save_table_range(tables, csv_path, range(1500, 1501), True)  # gittables
        # prepare_training(range(100, 1000), 50, False)
        # prepare_training(range(1000, 1500), 50, True)


def unique_columns(table_range: Iterable, cache_csv: bool, pretty: bool, do_print: bool, csv_path='test.csv') -> list[list]:
    """Compute all unique columns for a range of tables.

    Args:
        table_range (Iterable): a range of table IDs
        cache_csv (bool): if True, use csv files in the data/ folder or create them if they don't exist
        pretty (bool): if True, the result will contain table and column names, otherwise it will just contain the column ids
        do_print (bool): if True, the result will be printed to the command line
        csv_path (str, optional): If not None, the result will be saved as a csv under the given path. Defaults to 'test.csv'.

    Returns:
        list[list]: the result as a two dimensional list
    """
    max_rows = 10
    result = []
    counter = 0
    number_of_tables = len(table_range)
    for i in table_range:
        counter += 1
        print(
            f"Table Nr. {i} ({counter}/{number_of_tables})         ", end='\r')
        if cache_csv:
            table = csv_interface.get_table(tables, csv_path, i, max_rows)
        else:
            table = tables.get_table(i, max_rows)
        unique_columns = algorithm.find_unique_columns(table)
        if pretty:
            unique_columns = tables.pretty_columns(i, unique_columns)
        if len(unique_columns) > 1:
            result.append(unique_columns)
    sys.stdout.write("\033[K")

    if do_print:
        pprint(result)
    if csv_path is not None:
        if pretty:
            result = [tables.pretty_columns_header(), *result]
        arr = np.asarray(
            result, dtype=object)
        pd.DataFrame(arr).to_csv(csv_path, header=None, index=False)

    return result


def prepare_training(table_range: Iterable, number_rows: int, non_trivial: bool, path='src/data/training/'):
    if non_trivial:
        path = f"{path}{min(table_range)}-{max(table_range)}_{number_rows}_nt.csv"
    else:
        path = f"{path}{min(table_range)}-{max(table_range)}_{number_rows}.csv"
    path_result = path.replace(".csv", "-result.csv")
    ml = MachineLearning()
    na = NaiveAlgorithm()
    pd.DataFrame([], columns=ml.header).to_csv(path, index=False)
    pd.DataFrame([], columns=["PK Candidates"]).to_csv(
        path_result, index=False)
    for tableid in table_range:
        # TODO: error catching etc.
        table = csv_interface.get_table_local(csv_path, tableid, number_rows)
        data = ml.prepare_table(table)
        if non_trivial:
            # remove all trivial cases
            trivial_cases = data[data["Duplicates"] == 1].index
            data = data.drop(trivial_cases)
        data.to_csv(path, mode='a', header=False, index=False)
        data = na.find_unique_columns(table)
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


if __name__ == '__main__':
    main()
