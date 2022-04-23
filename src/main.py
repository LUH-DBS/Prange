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

from datasets import Gittables, Maintables, OpenData, CSV

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
        # unique_columns(range(100, 101), True, True, True)
        # save_csv(range(200, 2200)) # opendata
        # save_csv(range(100, 1000), True)  # gittables
        prepare_training(range(100, 200), 5)


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
    if cache_csv:
        csv_tables = CSV(get_csv_path())
    max_rows = 10
    result = []
    counter = 0
    number_of_tables = len(table_range)
    for i in table_range:
        counter += 1
        print(
            f"Table Nr. {i} ({counter}/{number_of_tables})         ", end='\r')
        if cache_csv:
            if exists(f"{get_csv_path()}{i}.csv"):
                table = csv_tables.get_table(i, max_rows)
            else:
                table = tables.get_table(i, max_rows)
                table.to_csv(f"{get_csv_path()}{i}.csv", index=False)
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


def get_csv_path() -> str:
    path = "src/data/"
    if isinstance(tables, Gittables):
        path += "gittables/"
    elif isinstance(tables, OpenData):
        path += "opendata/"
    elif isinstance(tables, Maintables):
        path += "maintables/"
    return path


def save_csv(table_range: Iterable, skip_existing=True, max_rows=-1) -> None:
    counter = 0
    number_of_tables = len(table_range)
    for tableid in table_range:
        counter += 1
        print(
            f"Saving table Nr. {tableid} ({counter}/{number_of_tables})         ", end='\r')
        if not skip_existing or not exists(f"{get_csv_path()}{tableid}.csv"):
            table = tables.get_table(tableid, max_rows)
            table.to_csv(f"{get_csv_path()}{tableid}.csv", index=False)
    sys.stdout.write("\033[K")
    print(
        f"Saved {number_of_tables} tables (from {table_range[0]} to {table_range[-1]})")


def prepare_training(table_range: Iterable, number_rows: int, path='src/data/training/'):
    path = f"{path}{min(table_range)}-{max(table_range)}.csv"
    ml = MachineLearning()
    csv = CSV(get_csv_path())
    pd.DataFrame([], columns=ml.header).to_csv(path)
    for tableid in table_range:
        # TODO: error catching etc.
        table = csv.get_table(tableid, number_rows)
        data = ml.prepare_table(table)
        data.to_csv(path, mode='a', header=False)


if __name__ == '__main__':
    main()
