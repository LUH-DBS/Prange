import pandas as pd
import numpy as np
from pprint import pprint
import sys
from typing import Iterable

import datasets.sql.csv_cache as csv_cache


def unique_columns(table_range: Iterable, csv_path: str) -> list[list]:
    """Compute all unique columns for a range of tables.

    Args:
        table_range (Iterable): a range of table IDs
        csv_path (str): The path to the input tables

    Returns:
        list[list]: the resulting list of tables with the IDs of unique columns
    """
    result = []
    counter = 0
    number_of_tables = len(table_range)
    for tableid in table_range:
        counter += 1
        print(
            f"Table Nr. {tableid} ({counter}/{number_of_tables})         ", end='\r')
        table = csv_cache.get_table_local(csv_path, tableid, -1)
        unique_columns = _find_unique_columns_table(table)
        if len(unique_columns) > 1:
            result.append(unique_columns)
    sys.stdout.write("\033[K")
    return result


def unique_columns_online(table_range: Iterable, csv_path: str, dataset) -> list:
    """Compute all unique columns for a range of tables.

    Args:
        table_range (Iterable): a range of table IDs
        csv_path (str): The path to the input tables
        dataset: an interface class of the dataset

    Returns:
        list: the resulting list of tables with the IDs of unique columns
    """
    result = []
    counter = 0
    number_of_tables = len(table_range)
    for i in table_range:
        counter += 1
        print(
            f"Table Nr. {i} ({counter}/{number_of_tables})         ", end='\r')
        table = dataset.get_table(i, -1)
        unique_columns = _find_unique_columns_table(table)
        if len(unique_columns) > 1:
            result.append(unique_columns)
    sys.stdout.write("\033[K")
    return result


def _find_unique_columns_table(table: pd.DataFrame) -> list:
    """Generate a list with all column ids which only contain unique values making use of sorting.

    Args:
        table (pd.Dataframe): the table to inspect

    Returns:
        list: the indexes of the unique columns
    """
    tablelength = len(table)
    nunique = table.nunique().values
    unique_columns = []
    for index, value in enumerate(nunique):
        if value == tablelength:
            unique_columns.append(index)
    return unique_columns
