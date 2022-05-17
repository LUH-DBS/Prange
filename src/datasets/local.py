"""Interface Module to read locally saved files."""
import os
from typing import Iterator

import pandas as pd
from pyarrow.parquet import ParquetFile
import pyarrow as pa


def traverse_directory_path(path: str, files_per_dir: int = -1, skip_tables: int = -1) -> Iterator[str]:
    """Returns an Iterator which iterates through local files returning their filepath.

    Args:
        path (str): The path where the files are located.
        nrows (int, optional): The number of rows to read from each file. Defaults to -1.
        files_per_dir (int, optional): The number of files to read from each (sub-)directory at max. Defaults to -1.
        skip_tables (int, optional): The number of tables/files to skip before returning them as DataFrames. Defaults to -1.

    Yields:
        str: The path to the file, if it has an extension of '.csv' or '.parquet'
    """
    skipcounter = 0
    for root, dirs, files in os.walk(path):
        filecounter = 0
        for file in files:
            if skip_tables > 0 and skipcounter < skip_tables:
                skipcounter += 1
                continue
            filecounter += 1
            if files_per_dir > 0 and filecounter > files_per_dir:
                break
            match os.path.splitext(file)[1]:
                case '.parquet':
                    yield f"{root}/{file}"
                case '.csv':
                    yield f"{root}/{file}"
                case _:
                    print(
                        f'file {file} with unsupported extension {os.path.splitext(file)[1]}')


def traverse_directory(path: str, nrows: int = -1, files_per_dir: int = -1, skip_tables: int = -1) -> Iterator[pd.DataFrame]:
    """Returns an Iterator which iterates through local files converted to DataFrames.

    Args:
        path (str): The path where the files are located.
        nrows (int, optional): The number of rows to read from each file. Defaults to -1.
        files_per_dir (int, optional): The number of files to read from each (sub-)directory at max. Defaults to -1.
        skip_tables (int, optional): The number of tables/files to skip before returning them as DataFrames. Defaults to -1.

    Yields:
        DataFrame: A Dataframe from the files und [path].
    """
    skipcounter = 0
    for root, dirs, files in os.walk(path):
        filecounter = 0
        for file in files:
            if skip_tables > 0 and skipcounter < skip_tables:
                skipcounter += 1
                continue
            filecounter += 1
            if files_per_dir > 0 and filecounter > files_per_dir:
                break
            try:
                match os.path.splitext(file)[1]:
                    case '.parquet':
                        yield get_table_from_parquet(f"{root}/{file}", nrows)
                    case '.csv':
                        yield get_table_from_csv(f"{root}/{file}", nrows)
                    case _:
                        print(
                            f'file {file} with unsupported extension {os.path.splitext(file)[1]}')
            except ValueError as error:
                print('Error:', error, f'({root}/{file})')
                continue


def get_table(path: str, nrows: int = -1) -> pd.DataFrame:
    match os.path.splitext(path)[1]:
        case '.parquet':
            return get_table_from_parquet(path, nrows)
        case '.csv':
            return get_table_from_csv(path, nrows)
        case _:
            pass


def get_table_from_parquet(path: str, nrows: int = -1) -> pd.DataFrame:
    """Read a parquet file as a DataFrame.

    Args:
        path (str): The path to the file.
        nrows (int, optional): The number of rows to read. Defaults to -1.

    Returns:
        pd.DataFrame: The table as a DataFrame.
    """
    if nrows > 0:
        try:
            file = ParquetFile(path)
            first_rows = next(file.iter_batches(batch_size=nrows))
            return pa.Table.from_batches([first_rows]).to_pandas()
        except StopIteration:
            return pd.read_parquet(path)
    else:
        return pd.read_parquet(path)


def get_table_from_csv(path: str, nrows: int = -1) -> pd.DataFrame:
    """Read a csv file as a DataFrame.

    Args:
        path (str): The path to the file.
        nrows (int, optional): The number of rows to read. Defaults to -1.

    Returns:
        pd.DataFrame: The table as a DataFrame.
    """
    if nrows > 0:
        return pd.read_csv(path, nrows=nrows)
    else:
        return pd.read_csv(path)
