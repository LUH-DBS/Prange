from multiprocessing import connection
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

    table = get_gittable(cursor, 989)

    import pandas as pd
    import numpy as np
    arr = np.asarray(table)
    pd.DataFrame(arr).to_csv('test.csv', header=None, index=False)

    connection.close()


def get_gittable(cursor, tableid):
    table = []
    table.append(get_gittable_columninfo(cursor, tableid))
    query = "SELECT count(distinct rowid) FROM gittables_main_tokenized WHERE tableid = %s"
    cursor.execute(query, (tableid,))
    row_count = cursor.fetchone()[0]

    # query = "SELECT count(distinct columnid) FROM gittables_main_tokenized WHERE tableid = %s"
    # cursor.execute(query, (tableid,))
    # column_count = cursor.fetchone()[0]

    query = "SELECT tokenized FROM gittables_main_tokenized WHERE tableid = %s and rowid = %s"
    for rowid in range(0, row_count):
        cursor.execute(query, (tableid, rowid))
        table.append([r[0] for r in cursor.fetchall()])
    return table


def get_gittable_columninfo(cursor, tableid):
    query = "SELECT header FROM gittables_columns_info WHERE tableid = %s"
    cursor.execute(query, (tableid,))
    return [r[0] for r in cursor.fetchall()]


if __name__ == '__main__':
    main()
