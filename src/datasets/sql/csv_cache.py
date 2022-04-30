from typing import Iterable
from ._sql_base import SQL_Baseclass
import pandas
from genericpath import exists
import sys
from pathlib import Path


def get_table_local(path: str, tableid: int, max_rows: int) -> pandas.DataFrame:
    """Return a complete table from the locally saved csv files with the id [tableid].

    Args:
        path (str): the path to the folder where the table is saved
        tableid (int): the id of the table
        max_rows (int): the maximum number of rows the returned table will have (only if flipped is False)

    Returns:
        pandas.DataFrame: the table as a DataFrame
    """
    try:
        if max_rows > 0:
            table = pandas.read_csv(f"{path}{tableid}.csv", nrows=max_rows)
        else:
            table = pandas.read_csv(f"{path}{tableid}.csv")
        return table
    except pandas.errors.EmptyDataError:
        print(f"The file {tableid} is empty.")
        return pandas.DataFrame([])


def get_table(dataset_interface: SQL_Baseclass, path: str, tableid: int, max_rows: int) -> pandas.DataFrame:
    """Return a complete table from the locally saved csv files with the id [tableid]
    or from the database if no local file exists.

    Args:
        path (str): the path to the folder where the table is saved
        tableid (int): the id of the table
        max_rows (int): the maximum number of rows the returned table will have (only if flipped is False)

    Returns:
        pandas.DataFrame: the table as a DataFrame
    """
    if exists(f"{path}{tableid}.csv"):
        table = get_table(tableid, max_rows)
    else:
        table = dataset_interface.get_table(tableid, max_rows)
        table.to_csv(f"{path}{tableid}.csv", index=False)
    return table


def save_table_range(dataset_interface: SQL_Baseclass, path: str, table_range: Iterable, skip_existing=True, max_rows=-1) -> None:
    counter = 0
    number_of_tables = len(table_range)
    for tableid in table_range:
        counter += 1
        print(
            f"Saving table Nr. {tableid} ({counter}/{number_of_tables})         ", end='\r')
        if not skip_existing or not exists(f"{path}{tableid}.csv"):
            table = dataset_interface.get_table(tableid, max_rows)
            Path(path).mkdir(parents=True, exist_ok=True)
            table.to_csv(f"{path}{tableid}.csv", index=False)
    sys.stdout.write("\033[K")
    print(
        f"Saved {number_of_tables} tables (from {table_range[0]} to {table_range[-1]})")
