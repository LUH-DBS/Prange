

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
