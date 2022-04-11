import os
from pprint import pprint
from dotenv import load_dotenv
from sqlalchemy import create_engine
import sys
from pprint import pprint
import pandas as pd
import numpy as np

from datasets import Gittables, Maintables, OpenData

from algorithms import NaiveAlgorithm

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
        algorithm = NaiveAlgorithm()
        unique_columns(200, 205, True, True)


def unique_columns(mintable: int, maxtable: int, pretty: bool, do_print: bool, csv_path='test.csv') -> list[list]:
    """Compute all unique columns for a range of tables.

    Args:
        mintable (int): the first table id (inclusive)
        maxtable (int): the last table id (exclusive)
        pretty (bool): if True, the result will contain table and column names, otherwise it will just contain the column ids
        do_print (bool): if True, the result will be printed to the command line
        csv_path (str, optional): If not None, the result will be saved as a csv under the given path. Defaults to 'test.csv'.

    Returns:
        list[list]: the result as a two dimensional list
    """
    result = []
    counter = 0
    number_of_tables = maxtable - mintable
    for i in range(mintable, maxtable):
        counter += 1
        print(
            f"Table Nr. {i} ({counter}/{number_of_tables})         ", end='\r')

        table = tables.get_table(i, -1)
        unique_columns = algorithm.find_unique_columns(table)
        if pretty:
            unique_columns = tables.pretty_columns(i, unique_columns)
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


################################################################################

if __name__ == '__main__':
    main()
