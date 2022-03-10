import os
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

    table = get_gittable(cursor, 989, True)
    # from pprint import pprint
    # pprint(table)

    print(find_unique_columns(table))

    import pandas as pd
    import numpy as np
    arr = np.asarray(table)
    pd.DataFrame(arr).to_csv('test.csv', header=None, index=False)

    connection.close()


def get_gittable(cursor, tableid: int, flipped: bool) -> list[list]:
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
        table.append(get_gittable_columninfo(cursor, tableid))
        query = "SELECT tokenized FROM gittables_main_tokenized WHERE tableid = %s and rowid = %s"
        for rowid in range(0, row_count):
            cursor.execute(query, (tableid, rowid))
            table.append([r[0] for r in cursor.fetchall()])
        return table


def get_gittable_columninfo(cursor, tableid: int) -> list:
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


def find_unique_columns(table: list[list]) -> list[int]:
    """Generate a list with all column ids which only contain unique values.

    Args:
        table (list[list]): the table which has to be flipped (a list of columns, not of rows)

    Returns:
        list[int]: the indexes of the unique columns
    """

    def column_is_unique(column: list) -> bool:
        """Determine wether a column contains only unique values.

        Args:
            column (list): the column to test

        Returns:
            bool: True if unique, False otherwise
        """
        for index in range(len(column)):
            if index == 0:
                continue
            if column[index] == column[index - 1]:  # TODO: what about NULL/NaN values?
                return False
        return True

    uniques = []
    for index, column in enumerate(table):
        column = sorted(column)  # TODO: implement own sorting/hashing?
        if column_is_unique(column):
            uniques.append(index)
    return uniques


if __name__ == '__main__':
    main()
