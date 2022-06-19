import os
from datetime import datetime
from typing import Iterable, List
from dotenv import load_dotenv
from pathlib import Path
from shutil import rmtree
import pandas as pd
from pyarrow.lib import ArrowInvalid
import re

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

BASE_PATH_DATA = 'src/data/'
TRAIN_DATASOURCE = 'gittables-parquet'
TEST_DATASOURCE = 'gittables-parquet'
RESULT_PATH_CORRECTNESS = "src/result/correctness"
SCORING_STRATEGIES = [
    [['recall', 'precision'], [recall, precision]],
]
MIN_ROWS = 100
MIN_COLS = 3
TRAIN_TABLE_COUNT = 10000
TEST_TABLE_COUNT = 1000


def main():
    TRAIN_MODEL = True  # ! training has to be seperate, otherwise logging will not work
    if TRAIN_MODEL:
        setup_logging(log_to_file=False, level=logging.DEBUG)
        testing.prepare_and_train(row_count_iter=MODEL_ROWS_LIST,
                                  train_table_count=TRAIN_TABLE_COUNT,
                                  data_path=BASE_PATH_DATA + TRAIN_DATASOURCE,
                                  train_envenly=False,
                                  scoring_strategies=SCORING_STRATEGIES,
                                  train_time=TRAIN_TIME,
                                  min_rows=MIN_ROWS,
                                  min_cols=MIN_COLS
                                  )
    else:
        setup_logging(log_to_file=True, level=logging.INFO)
        # download_dataset()
        # dataset_info()
        # logger.setLevel(logging.DEBUG)
        # speed_test()
        # logger.setLevel(logging.INFO)
        # speed_test_to_tex()
        correctness_test(log_false_guesses=True)
        # correctness_test_to_tex()


def compare_input_sizes(train_model: bool = False):
    model_rows_list = [5, 10, 20, 50]
    train_table_count = TRAIN_TABLE_COUNT
    test_table_count = TEST_TABLE_COUNT  # -1 is as much as possible
    train_time = [5 * 60 * 60]  # 5 hours
    if train_model:
        setup_logging(log_to_file=False, level=logging.DEBUG)
        testing.prepare_and_train(row_count_iter=model_rows_list,
                                  train_table_count=train_table_count,
                                  data_path=BASE_PATH_DATA + TRAIN_DATASOURCE,
                                  train_envenly=False,
                                  scoring_strategies=SCORING_STRATEGIES,
                                  train_time_list=train_time,
                                  min_rows=MIN_ROWS,
                                  min_cols=MIN_COLS
                                  )
    else:
        rmtree(f'{RESULT_PATH_CORRECTNESS}/long/compare-input-sizes',
               ignore_errors=True)
        setup_logging(log_to_file=True, level=logging.INFO)
        correctness_test(experiment_name='compare-input-sizes',
                         model_rows_list=model_rows_list,
                         train_table_count=train_table_count,
                         test_table_count=test_table_count,
                         train_time_list=train_time,
                         skip_tables=train_table_count,
                         result_file_name="{nrows}rows.csv",
                         log_false_guesses=True)
        # correctness_test_to_tex()


def compare_training_time(train_model: bool = False):
    model_rows_list = [10]
    train_table_count = TRAIN_TABLE_COUNT
    test_table_count = TEST_TABLE_COUNT  # -1 is as much as possible
    train_time = [5*60, 10*60, 20*60, 40*60,
                  1*60*60, 2*60*60, 3*60*60, 5*60*60]
    if train_model:
        setup_logging(log_to_file=False, level=logging.DEBUG)
        testing.prepare_and_train(row_count_iter=model_rows_list,
                                  train_table_count=train_table_count,
                                  data_path=BASE_PATH_DATA + TRAIN_DATASOURCE,
                                  train_envenly=False,
                                  scoring_strategies=SCORING_STRATEGIES,
                                  train_time_list=train_time,
                                  min_rows=MIN_ROWS,
                                  min_cols=MIN_COLS
                                  )
    else:
        rmtree(f'{RESULT_PATH_CORRECTNESS}/long/compare-training-time',
               ignore_errors=True)
        setup_logging(log_to_file=True, level=logging.INFO)
        correctness_test(experiment_name='compare-training-time',
                         model_rows_list=model_rows_list,
                         train_table_count=train_table_count,
                         test_table_count=test_table_count,
                         train_time_list=train_time,
                         skip_tables=train_table_count,
                         result_file_name="{train_time}minutes.csv",
                         log_false_guesses=True)
        # correctness_test_to_tex()


def speed_test(train_model: bool = False):
    logger.setLevel(logging.DEBUG)
    model_rows_list = [5, 10, 20]
    train_table_count = TRAIN_TABLE_COUNT
    train_time_list = [3 * 60 * 60]
    ROW_NUMBER = 100000000
    COL_NUMBER = 10
    logger.info("Starting speed test")
    percentages = [60, 70, 80, 90]
    if train_model:
        setup_logging(log_to_file=False, level=logging.DEBUG)
        testing.prepare_and_train(row_count_iter=model_rows_list,
                                  train_table_count=train_table_count,
                                  data_path=BASE_PATH_DATA + TRAIN_DATASOURCE,
                                  train_envenly=False,
                                  scoring_strategies=SCORING_STRATEGIES,
                                  train_time_list=train_time_list,
                                  min_rows=MIN_ROWS,
                                  min_cols=MIN_COLS
                                  )
    else:
        for scoring_strategy_name, scoring_strategy in SCORING_STRATEGIES:
            for train_time in train_time_list:
                for model in model_rows_list:
                    for percentage in percentages:
                        # parquet, load only whats necessary
                        random_int(max_row_size=ROW_NUMBER,
                                   csv=False,
                                   use_small_tables=True,
                                   generate_tables=True,
                                   nunique_percent=percentage,
                                   rows_model=model,
                                   ncols=COL_NUMBER,
                                   path_to_model=f'src/data/model/{model}_rows/{train_table_count}_tables/{TRAIN_DATASOURCE}/{int(train_time/60)}minutes/{"_".join(scoring_strategy_name)}.pickle'
                                   )
                        # parquet, load everything
                        random_int(max_row_size=ROW_NUMBER,
                                   csv=False,
                                   use_small_tables=False,
                                   generate_tables=False,
                                   nunique_percent=percentage,
                                   rows_model=model,
                                   ncols=COL_NUMBER,
                                   path_to_model=f'src/data/model/{model}_rows/{train_table_count}_tables/{TRAIN_DATASOURCE}/{int(train_time/60)}minutes/{"_".join(scoring_strategy_name)}.pickle'
                                   )
                        # csv
                        random_int(max_row_size=ROW_NUMBER,
                                   csv=True,
                                   use_small_tables=False,
                                   generate_tables=True,
                                   nunique_percent=percentage,
                                   rows_model=model,
                                   ncols=COL_NUMBER,
                                   path_to_model=f'src/data/model/{model}_rows/{train_table_count}_tables/{TRAIN_DATASOURCE}/{int(train_time/60)}minutes/{"_".join(scoring_strategy_name)}.pickle'
                                   )
    logger.setLevel(logging.INFO)
    speed_test_to_tex()


def correctness_test(experiment_name: str, model_rows_list: List[int], train_table_count: int, test_table_count: int, train_time_list: List[int], skip_tables: int, result_file_name: str, log_false_guesses: bool = False):
    """Train and test models which look at nrows rows for their prediction.

    Args:
        nrows_iter (Iterable[int]): A model will be trained/tested for each item in the Iterable.
        train_model (bool, optional): Only train the models if True. Defaults to False.
    """
    logger.info("Started the correctness test")
    if log_false_guesses:
        rmtree(f'{RESULT_PATH_CORRECTNESS}/false_pos', ignore_errors=True)
        rmtree(f'{RESULT_PATH_CORRECTNESS}/false_neg', ignore_errors=True)
    for strategy in SCORING_STRATEGIES:
        strategy = '_'.join(strategy[0])
        for nrows in model_rows_list:
            for train_time in train_time_list:
                logger.info(
                    f"Correctness Testing for '{experiment_name}' with a {nrows}rows model with strategy {strategy} ({int(train_time / 60)}minutes train time)")
                result_file_path = f"{RESULT_PATH_CORRECTNESS}/long/{experiment_name}/{result_file_name.format(strategy=strategy, nrows=nrows, train_time=int(train_time / 60))}"
                Path(result_file_path.rsplit('/', 1)
                     [0]).mkdir(parents=True, exist_ok=True)
                testing.test_model(path_to_model=f'src/data/model/{nrows}_rows/{train_table_count}_tables/{TRAIN_DATASOURCE}/{int(train_time / 60)}minutes/{strategy}.pickle',
                                   model_rows=nrows,
                                   input_path=BASE_PATH_DATA + TEST_DATASOURCE,
                                   output_path=result_file_path,
                                   files_per_dir=-1,
                                   skip_tables=skip_tables,
                                   use_small_tables=False,
                                   speed_test=False,
                                   max_files=test_table_count,
                                   min_rows=MIN_ROWS,
                                   min_cols=MIN_COLS,
                                   log_false_guesses=log_false_guesses
                                   )
    testing.correctness_summary(input_dir=f"{RESULT_PATH_CORRECTNESS}/long/{experiment_name}",
                                output_file=f'{RESULT_PATH_CORRECTNESS}/summary/{experiment_name}.csv')
    logger.info(f"Finished correctness test for '{experiment_name}'")


def random_int(max_row_size: int, path_to_model: str, generate_tables: bool = True, use_small_tables: bool = True, csv: bool = False, nunique_percent: int = 0, ncols: int = 10, rows_model: int = 10):
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
                            path_to_model=path_to_model,
                            model_rows=rows_model,
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


def get_example_features(max_count: int = 2, table_count: int = 1000, out_path: str = 'features.csv'):
    pd.DataFrame([], columns=machine_learning.header).to_csv(
        out_path, index=False)
    datatype_dict = {}
    counter = 0
    for table in local.traverse_directory("src/data/gittables-parquet"):
        counter += 1
        if counter > table_count:
            break
        prepared = machine_learning.prepare_table(table)
        result = pd.DataFrame()
        for row in prepared.values:
            dt = row[1]
            if dt in datatype_dict:
                if datatype_dict[dt] >= max_count:
                    continue
                else:
                    datatype_dict[dt] += 1
                    result = pd.concat([result, pd.DataFrame([row])])
            else:
                datatype_dict[dt] = 1
                result = pd.concat([result, pd.DataFrame([row])])
        result.to_csv(out_path, index=False, mode='a', header=False)


def speed_test_to_tex(input_path: str = 'src/result/speed_random-int',):
    tmp_path = '/tmp/thesis-tmp'
    for table_path in local.traverse_directory_path(input_path):
        if 'small-tables' in table_path:
            keep_columns = ['Rows', 'ML: Loading I', 'ML: Compute Time', 'ML: Loading II',
                            'ML: Validation Time', 'ML: Total', 'Naive: Loading', 'Naive: Compute Time', 'Naive: Total']
            small_table = True
        else:
            keep_columns = ['Rows', 'ML: Loading I', 'ML: Compute Time', 'ML: Validation Time',
                            'ML: Total', 'Naive: Loading', 'Naive: Compute Time', 'Naive: Total']
            small_table = False
        table = local.get_table(table_path)
        keep_table = table[keep_columns]
        if not small_table:
            keep_table = keep_table.rename(
                columns={'ML: Loading I': 'ML: Loading'})
        keep_table = keep_table.sort_values(["Rows"])
        tmp_file = table_path.replace(input_path, tmp_path)
        Path(tmp_file.rsplit('/', 1)[0]).mkdir(parents=True, exist_ok=True)
        keep_table.to_csv(tmp_file, index=False)
    csv_to_tex(input_path=tmp_path,
               output_path='main/table-code/result/efficiency',
               align='S',
               )
    rmtree(tmp_path)


def csv_to_tex(input_path: str = 'src/result/speed_random-int', output_path: str = 'main/table-code/result/efficiency', align: str = 'c'):
    from tably import Tably, AttrDict
    options = {
        'outfile': None,
        'no_indent': False,
        'skip': 0,
        'no_header': False,
        'preamble': False,
        'sep': ',',
        'units': None,
        'no_escape': True,
        'fragment': False,
        'fragment_skip_header': False,
        'replace': True
    }
    for root, dirs, files in os.walk(input_path):
        for file in files:
            filepath = root + '/' + file
            file_format = 'csv'
            small_tables = False
            match file:
                case x if 'parquet' in x:
                    file_format = 'parquet'
                case x if 'small-tables' in x:
                    small_tables = True
            tmp = re.findall("\d+rowModel", filepath)
            tmp = [re.findall("\d+", x) for x in tmp]
            if tmp and tmp[0]:
                model_rows = tmp[0][0]
            tmp = re.findall("\d+colTable", filepath)
            tmp = [re.findall("\d+", x) for x in tmp]
            if tmp and tmp[0]:
                table_cols = tmp[0][0]
            tmp = re.findall("\d+percent", filepath)
            tmp = [re.findall("\d+", x) for x in tmp]
            if tmp and tmp[0]:
                nonunique_percent = tmp[0][0]
            caption = f'The result of the efficiency test with a generated table with \SI{{{100-int(nonunique_percent)}}}{{\\percent}} unique columns in a {file_format} file format. The test was conducted on a model with an input size of {model_rows} rows on tables with {table_cols} columns.'
            if small_tables:
                caption += ' In this experiment, only the necessary rows were loaded in a first and only the necessary columns in a second step.'
            label = f'table:efficiency_{file.replace(".csv", "")}'
            outdir = root.replace(input_path, output_path)
            Path(outdir).mkdir(parents=True, exist_ok=True)
            args = AttrDict(**options, files=[filepath],
                            separate_outfiles=[outdir], align=align, caption=caption, label=label)
            Tably(args).run()


if __name__ == '__main__':
    main()
