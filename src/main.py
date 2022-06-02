import os
from datetime import datetime
from typing import Iterable
from dotenv import load_dotenv
from pathlib import Path
from shutil import rmtree
import pandas as pd

from autosklearn.metrics import accuracy, precision, recall, f1

import datasets.local as local
from datasets.sql import Gittables, Maintables, OpenData
import datasets.sql.csv_cache as csv_cache
import algorithms.naive_algorithm as naive_algorithm
import algorithms.machine_learning as machine_learning
import testing

load_dotenv()
db_params = {
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
}


def main():
    # download_dataset(csv=True)
    testcase_1(nrows_iter=[5, 10, 20], test_table_count=1000)


def testcase_1(nrows_iter: Iterable[int], test_table_count: int, train_model: bool = False):
    """Train and test models which look at nrows rows for their prediction.

    Args:
        nrows_iter (Iterable[int]): A model will be trained/tested for each item in the Iterable.
        train_model (bool, optional): Only train the models if True. Defaults to False.
    """
    scoring_strategies = [
        [['recall', 'precision'], [recall, precision]],
        # [['recall', 'recall', 'precision'], [recall, recall, precision]]
    ]

    train_table_count = 10000
    train_datasource = 'gittables'
    test_datasource = 'gittables/abstraction_tables_licensed'
    train_time = 10800  # 3 hours
    result_path = "src/result"
    result_path_long = f"{result_path}/long/{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
    Path(result_path_long).mkdir(parents=True, exist_ok=True)

    if train_model:
        testing.prepare_and_train(row_count_iter=nrows_iter,
                                  train_table_count=train_table_count,
                                  data_path=f'src/data/{train_datasource}',
                                  train_envenly=False,
                                  scoring_strategies=scoring_strategies,
                                  train_time=train_time)
    for nrows in nrows_iter:
        testing.test_model(path_to_model=f'src/data/model/{nrows}_rows/{train_table_count}_tables/{train_datasource}/{int(train_time / 60)}minutes/recall_precision.pickle',
                           nrows=nrows,
                           input_path=f'src/data/{test_datasource}/',
                           output_path=f'{result_path_long}/{nrows}rows.csv',
                           files_per_dir=100,
                           skip_tables=-1)


def random_int():
    testing.test_random_int(row_counts=[100, 1000, 10000, 100000, 1000000, 5000000, 10000000],
                            ncols=100,
                            out_path="test-random-int.csv",
                            path_to_model='src/data/model/10_rows/10000_tables/gittables/180minutes/recall_precision.pickle',
                            nrows=10)


def download_dataset(csv: bool = True):
    import requests

    ACCESS_TOKEN = os.getenv("ZENODO_API_KEY")
    if csv:
        record_id = "6515973"  # CSV
        filepath_base = "src/data/gittables-csv/"
    else:
        record_id = "6517052"  # parquet
        filepath_base = "src/data/gittables-parquet/"
    Path(filepath_base).mkdir(parents=True, exist_ok=True)
    Path('tmp').mkdir(parents=True, exist_ok=True)

    r = requests.get(
        f"https://zenodo.org/api/records/{record_id}", params={'access_token': ACCESS_TOKEN})
    download_urls = [f['links']['self'] for f in r.json()['files']]
    filenames = [f['key'] for f in r.json()['files']]

    print(f"Downloading {len(download_urls)} folders.")
    # print(download_urls)
    # print(filenames)

    for filename, url in zip(filenames, download_urls):
        print("Downloading:", filename)
        r = requests.get(url, params={'access_token': ACCESS_TOKEN})
        from zipfile import ZipFile
        with open('tmp/' + filename, 'wb') as f:
            f.write(r.content)
        with ZipFile('tmp/' + filename, 'r') as zip_ref:
            zip_ref.extractall(filepath_base + filename.replace(".zip", ""))
    rmtree('tmp')


if __name__ == '__main__':
    main()
