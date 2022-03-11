import os
from unittest import result
from dotenv import load_dotenv
import psycopg2

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

    unique_columns(cursor, 2000, 2050, False)

    connection.close()


def unique_columns(cursor, mintable: int, maxtable: int, do_print: bool, do_csv='test.csv') -> list[list]:
    """Compute all unique columns for a range of tables.

    The result is a two dimensional list with the format ['tableid', 'tablename', 'columnids', 'columnnames']

    Args:
        cursor: _description_
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
        tablename = get_tablename_gittable(cursor, i)
        print(f"{counter}/{number_of_tables}:\t{tablename}         ", end='\r')

        table = get_table_gittable(cursor, i, True)
        unique_columns = find_unique_columns(table, 'hash')
        result.append([i,
                       tablename,
                       unique_columns,
                       get_columnnames_gittable(cursor, i, unique_columns)])

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


def get_table_gittable(cursor, tableid: int, flipped: bool) -> list[list]:
    """Return a complete table from the gittables with the id [tableid].

    Args:
        cursor: a cursor instance of the psycopg2 connection
        tableid (int): the id of the table
        flipped (bool): if True, a list of columns will be returned, otherwise a list of rows

    Returns:
        list[list]: the table in a list format
    """
    table = []
    query = "SELECT num_rows, num_columns FROM gittables_tables_info WHERE id = %s"
    cursor.execute(query, (tableid,))
    row_count, column_count = cursor.fetchone()

    if flipped:
        query = "SELECT tokenized FROM gittables_main_tokenized WHERE tableid = %s and columnid = %s"
        for columnid in range(0, column_count):
            cursor.execute(query, (tableid, columnid))
            table.append([r[0] for r in cursor.fetchall()])
        return table
    else:
        table.append(get_columninfo_gittable(cursor, tableid))
        query = "SELECT tokenized FROM gittables_main_tokenized WHERE tableid = %s and rowid = %s"
        for rowid in range(0, row_count):
            cursor.execute(query, (tableid, rowid))
            table.append([r[0] for r in cursor.fetchall()])
        return table


def get_columninfo_gittable(cursor, tableid: int) -> list:
    """Get the column header for a table.

    Args:
        cursor: a cursor instance of the psycopg2 connection
        tableid (int): the id of the table

    Returns:
        list: the header
    """
    query = "SELECT header FROM gittables_columns_info WHERE tableid = %s"
    cursor.execute(query, (tableid,))
    return [r[0] for r in cursor.fetchall()]


def get_tablename_gittable(cursor, tableid: int) -> str:
    """Get the name of a table.

    Args:
        cursor: a cursor instance of the psycopg2 connection
        tableid (int): the id of the table

    Returns:
        str: the name of the table
    """
    query = "SELECT filename FROM gittables_tables_info WHERE id = %s"
    cursor.execute(query, (tableid,))
    return cursor.fetchone()[0]


def get_columnnames_gittable(cursor, tableid: int, columnids: list[int]) -> list[str]:
    names = get_columninfo_gittable(cursor, tableid)
    result = []
    for i in columnids:
        result.append(names[i])
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

    match algorithm:  # type: ignore
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
