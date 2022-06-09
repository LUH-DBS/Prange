import os
from datetime import datetime
from typing import Iterable
from dotenv import load_dotenv
from pathlib import Path
from shutil import rmtree
import pandas as pd
from pyarrow.lib import ArrowInvalid

from autosklearn.metrics import accuracy, precision, recall, f1

import datasets.local as local
from datasets.sql import Gittables, Maintables, OpenData
import datasets.sql.csv_cache as csv_cache
import algorithms.naive_algorithm as naive_algorithm
import algorithms.machine_learning as machine_learning
import testing
import logging
logger = logging.getLogger(__name__)

load_dotenv()
db_params = {
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
}


def main():
    setup_logging()
    # download_dataset()
    dataset_info()
    random_int(max_row_size=100000000, csv=False,
               use_small_tables=True)
    random_int(max_row_size=100000000, csv=False,
               use_small_tables=False, generate_tables=False)
    random_int(max_row_size=100000000, csv=True,
               use_small_tables=False, generate_tables=False)
    # testcase_1(nrows_iter=[5, 10, 20], test_table_count=1000)
def speed_test():
    ROW_NUMBER = 100000000
    COL_NUMBER = 10
    logger.info("Starting speed test")
    percentages = [60, 70, 80, 90]
    for percentage in percentages:
        # parquet, load only whats necessary
        random_int(max_row_size=ROW_NUMBER,
                   csv=False,
                   use_small_tables=True,
                   generate_tables=True,
                   nunique_percent=percentage,
                   rows_model=10,
                   ncols=COL_NUMBER
                   )
        # parquet, load everything
        random_int(max_row_size=ROW_NUMBER,
                   csv=False,
                   use_small_tables=False,
                   generate_tables=False,
                   nunique_percent=percentage,
                   rows_model=10,
                   ncols=COL_NUMBER
                   )
        # csv
        random_int(max_row_size=ROW_NUMBER,
                   csv=True,
                   use_small_tables=False,
                   generate_tables=True,
                   nunique_percent=percentage,
                   rows_model=10,
                   ncols=COL_NUMBER
                   )
    for model in [5, 20]:
        random_int(max_row_size=ROW_NUMBER,
                   csv=False,
                   use_small_tables=True,
                   generate_tables=True,
                   nunique_percent=70,
                   rows_model=model,
                   ncols=COL_NUMBER
                   )


def testcase_1(nrows_iter: Iterable[int], test_table_count: int, train_model: bool = False):
    """Train and test models which look at nrows rows for their prediction.

    Args:
        nrows_iter (Iterable[int]): A model will be trained/tested for each item in the Iterable.
        train_model (bool, optional): Only train the models if True. Defaults to False.
    """
    logger.info("Started Testcase 1")
    scoring_strategies = [
        [['recall', 'precision'], [recall, precision]],
        # [['recall', 'recall', 'precision'], [recall, recall, precision]]
    ]

    train_table_count = 10000
    train_datasource = 'gittables'
    test_datasource = 'gittables-parquet'
    train_time = 10800  # 3 hours
    result_path = "src/result"
    result_path_long = f"{result_path}/long/{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
    Path(result_path_long).mkdir(parents=True, exist_ok=True)

    if train_model:
        logger.info("Started training the models")
        testing.prepare_and_train(row_count_iter=nrows_iter,
                                  train_table_count=train_table_count,
                                  data_path=f'src/data/{train_datasource}',
                                  train_envenly=False,
                                  scoring_strategies=scoring_strategies,
                                  train_time=train_time)
    for nrows in nrows_iter:
        logger.info("Testing model with %s rows", nrows)
        testing.test_model(path_to_model=f'src/data/model/{nrows}_rows/{train_table_count}_tables/{train_datasource}/{int(train_time / 60)}minutes/recall_precision.pickle',
                           nrows=nrows,
                           input_path=f'src/data/{test_datasource}/',
                           output_path=f'{result_path_long}/{nrows}rows.csv',
                           files_per_dir=10,
                           skip_tables=-1,
                           use_small_tables=True,
                           speed_test=False)
    logger.info("Finished Testcase 1")


def random_int(max_row_size: int, generate_tables: bool = True, use_small_tables: bool = True, csv: bool = False, nunique_percent: int = 0, ncols: int = 10, rows_model: int = 10):
    if csv:
        filetype = 'csv'
        use_small_tables = False
    else:
        filetype = 'parquet'
    logger.info(
        f"Started random_int test with filetype {filetype} and a maximum of {max_row_size:,d} rows (small_table={use_small_tables}, {nunique_percent}% nuniques)")
    row_list = [100, 1000, 10000, 100000, 1000000,
                5000000, 10000000, 50000000, 100000000]
    out_path = f"src/result/speed_random-int/{rows_model}rowModel-{ncols}colTable/"
    Path(out_path).mkdir(parents=True, exist_ok=True)
    testing.test_random_int(row_counts=[x for x in row_list if x <= max_row_size],
                            ncols=ncols,
                            out_path=f"{out_path}{filetype}-{nunique_percent}percent.csv",
                            path_to_model='src/data/model/10_rows/10000_tables/gittables/180minutes/recall_precision.pickle',
                            model_rows=rows_model,
                            nrows=10,
                            use_small_tables=use_small_tables,
                            generate_tables=generate_tables,
                            csv=csv,
                            nonunique_percent=nunique_percent
                            )
    logger.info("Finished random_int test")


def download_dataset(csv: bool = True):
    import requests
    logger.info("Started downloading the dataset")

    ACCESS_TOKEN = os.getenv("ZENODO_API_KEY")
    if csv:
        logger.info("Downloading csv files")
        record_id = "6515973"  # CSV
        filepath_base = "src/data/gittables-csv/"
    else:
        logger.info("Downloading parquet files")
        record_id = "6517052"  # parquet
        filepath_base = "src/data/gittables-parquet/"
    Path(filepath_base).mkdir(parents=True, exist_ok=True)
    Path('tmp').mkdir(parents=True, exist_ok=True)

    r = requests.get(
        f"https://zenodo.org/api/records/{record_id}", params={'access_token': ACCESS_TOKEN})
    download_urls = [f['links']['self'] for f in r.json()['files']]
    filenames = [f['key'] for f in r.json()['files']]

    logger.info("Downloading %s folders", len(download_urls))

    counter = 0
    for filename, url in zip(filenames, download_urls):
        counter += 1
        logger.info("Downloading: %s (%d/%d)", filename,
                    counter, len(download_urls))
        r = requests.get(url, params={'access_token': ACCESS_TOKEN})
        from zipfile import ZipFile
        with open('tmp/' + filename, 'wb') as f:
            f.write(r.content)
        with ZipFile('tmp/' + filename, 'r') as zip_ref:
            zip_ref.extractall(filepath_base + filename.replace(".zip", ""))
    rmtree('tmp')
    logger.info("Finished download")


def setup_logging(log_to_file: bool = True, level=logging.DEBUG):
    def addLoggingLevel(levelName, levelNum, methodName=None):
        """
        Comprehensively adds a new logging level to the `logging` module and the
        currently configured logging class.

        `levelName` becomes an attribute of the `logging` module with the value
        `levelNum`. `methodName` becomes a convenience method for both `logging`
        itself and the class returned by `logging.getLoggerClass()` (usually just
        `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
        used.

        To avoid accidental clobberings of existing attributes, this method will
        raise an `AttributeError` if the level name is already an attribute of the
        `logging` module or if the method name is already present 

        Example
        -------
        >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
        >>> logging.getLogger(__name__).setLevel("TRACE")
        >>> logging.getLogger(__name__).trace('that worked')
        >>> logging.trace('so did this')
        >>> logging.TRACE
        5

        """
        if not methodName:
            methodName = levelName.lower()

        if hasattr(logging, levelName):
            raise AttributeError(
                '{} already defined in logging module'.format(levelName))
        if hasattr(logging, methodName):
            raise AttributeError(
                '{} already defined in logging module'.format(methodName))
        if hasattr(logging.getLoggerClass(), methodName):
            raise AttributeError(
                '{} already defined in logger class'.format(methodName))

        # This method was inspired by the answers to Stack Overflow post
        # http://stackoverflow.com/q/2183233/2988730, especially
        # http://stackoverflow.com/a/13638084/2988730
        def logForLevel(self, message, *args, **kwargs):
            if self.isEnabledFor(levelNum):
                self._log(levelNum, message, args, **kwargs)

        def logToRoot(message, *args, **kwargs):
            logging.log(levelNum, message, *args, **kwargs)

        logging.addLevelName(levelNum, levelName)
        setattr(logging, levelName, levelNum)
        setattr(logging.getLoggerClass(), methodName, logForLevel)
        setattr(logging, methodName, logToRoot)

    addLoggingLevel("COMMON_ERROR", logging.DEBUG + 5)
    if log_to_file:
        Path("src/log/").mkdir(parents=True, exist_ok=True)
        log_path = f"src/log/{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}.csv"
        with open(log_path, "w") as file:
            file.write("Module,Function,Level,Time,Message\n")
        logging.basicConfig(
            level=level,
            filename=log_path,
            encoding='utf-8',
            format='%(name)s,%(funcName)s,%(levelname)s,%(asctime)s,"%(message)s"',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        logging.basicConfig(
            level=level,
            format='%(levelname)s (%(name)s:%(lineno)d): %(message)s'
        )


def dataset_info():
    logger.info("Starting to gather dataset info")
    MIN_COLS = 10
    over_100 = 0
    over_1000 = 0
    for dataset in ["gittables-parquet", "gittables-csv"]:
        counter = 0
        with open(f'dataset_info-{dataset}.csv', 'w') as file:
            # file.write("Folder,File,Rows,Columns\n")
            for path in local.traverse_directory_path(f'src/data/{dataset}/'):
                counter += 1
                # print(f"Table {counter}             ", end="\r")
                try:
                    table = local.get_table(path)
                except pd.errors.ParserError as e:
                    counter -= 1
                    logger.common_error(
                        "ParserError with file %s", path)
                    continue
                except ArrowInvalid as error:
                    counter -= 1
                    logger.common_error(
                        "ArrowInvalid error with file %s", path)
                    continue
                except UnicodeDecodeError:
                    counter -= 1
                    logger.common_error(
                        "UnicodeDecodeError with file %s", path)
                    continue
                if len(table.columns) >= MIN_COLS and len(table) > 100:
                    if len(table) > 1000:
                        over_1000 += 1
                    elif len(table) > 100:
                        over_100 += 1
                    row = [path.rsplit('/', 2)[1], path.rsplit('/', 1)
                           [1], len(table), len(table.columns)]
                    row = [str(x) for x in row]
                    # file.write(",".join(row) + "\n")
        logger.info(
            f"{dataset} has {over_100} tables with > 100 rows and {over_1000} tables with > 1000 rows")


if __name__ == '__main__':
    main()
