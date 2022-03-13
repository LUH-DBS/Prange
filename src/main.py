import os
from dotenv import load_dotenv
import psycopg2

from datasets import *

load_dotenv()
db_params = {
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
}


def main():
    connection = psycopg2.connect(**db_params)
    cursor = connection.cursor()

    global tables
    tables = Gittables(cursor)

    unique_columns(2000, 2050, False)

    connection.close()


def unique_columns(mintable: int, maxtable: int, do_print: bool, do_csv='test.csv') -> list[list]:
    """Compute all unique columns for a range of tables.

    The result is a two dimensional list with the format ['tableid', 'tablename', 'columnids', 'columnnames']

    Args:
        mintable (int): the first table id (inclusive)
        maxtable (int): the last table id (exclusive)
        do_print (bool): if True, the result will be printed to the command line
        do_csv (str, optional): If not None, the result will be saved as a csv under the given path. Defaults to 'test.csv'.

    Returns:
        list[list]: the result as a two dimensional list
    """
    result = []
    counter = 0
    number_of_tables = maxtable - mintable
    for i in range(mintable, maxtable):
        counter += 1
        tablename = tables.get_tablename_gittable(i)
        print(f"{counter}/{number_of_tables}:\t{tablename}         ", end='\r')

        table = tables.get_table_gittable(i, True)
        unique_columns = find_unique_columns(table, 'hash')
        result.append([i,
                       tablename,
                       unique_columns,
                       tables.get_columnnames_gittable(i, unique_columns)])

    if do_print:
        from pprint import pprint
        pprint(result)
    if do_csv is not None:
        if type(do_csv) == 'bool':
            do_csv = 'test.csv'
        import pandas as pd
        import numpy as np
        arr = np.asarray(
            [['tableid', 'tablename', 'columnids', 'columnnames'], *result], dtype=object)
        pd.DataFrame(arr).to_csv(do_csv, header=None, index=False)

    return result


################################################################################


def find_unique_columns(table: list[list], algorithm: str) -> list[int]:
    """Generate a list with all column ids which only contain unique values making use of sorting.

    Args:
        table (list[list]): the table which has to be flipped (a list of columns, not of rows)
        algorithm (str): either 'hash' or 'sort', raises an error otherwise

    Raises:
        ValueError: if algorithm is neither 'hash' nor 'sort'

    Returns:
        list[int]: the indexes of the unique columns
    """

    def column_is_unique_sorting(column: list) -> bool:
        """Determine wether a column contains only unique values.

        Args:
            column (list): the column to test

        Returns:
            bool: True if unique, False otherwise
        """
        column = sorted(column)  # TODO: implement own sorting/hashing?
        for index in range(len(column)):
            if index == 0:
                continue
            if column[index] == column[index - 1]:  # TODO: what about NULL/NaN values?
                return False
        return True

    def column_is_unique_hashing(column: list) -> bool:
        """Determine wether a column contains only unique values.

        Args:
            column (list): the column to test

        Returns:
            bool: True if unique, False otherwise
        """
        hashmap = {}
        for value in column:
            if value not in hashmap:
                hashmap[value] = 0
            else:
                return False
        return True

    match algorithm:
        case 'hash':
            column_is_unique = column_is_unique_hashing
        case 'sort':
            column_is_unique = column_is_unique_sorting
        case _:
            raise ValueError("Only 'hash' and 'sort' are valid algorithms.")

    uniques = []
    for index, column in enumerate(table):
        if column_is_unique(column):
            uniques.append(index)
    return uniques


if __name__ == '__main__':
    main()
