import pandas as pd
from pyarrow.parquet import ParquetFile
import pyarrow as pa
import os

from typing import Iterator


def traverse_directory(path: str, nrows=-1, files_per_dir=-1) -> Iterator:
    for root, dirs, files in os.walk(path):
        filecounter = 0
        for file in files:
            filecounter += 1
            if files_per_dir > 0 and filecounter > files_per_dir:
                break
            match os.path.splitext(file)[1]:
                case '.parquet':
                    yield get_table_from_parquet(f"{root}/{file}", nrows)
                case '.csv':
                    yield get_table_from_csv(f"{root}/{file}", nrows)
                case _:
                    print(
                        f'file {file} with unsupported extension {os.path.splitext(file)[1]}')


def get_table_from_parquet(path: str, nrows=-1) -> pd.DataFrame:
    if nrows > 0:
        try:
            file = ParquetFile(path)
            first_rows = next(file.iter_batches(batch_size=nrows))
            return pa.Table.from_batches([first_rows]).to_pandas()
        except StopIteration:
            return pd.read_parquet(path)
    else:
        return pd.read_parquet(path)


def get_table_from_csv(path: str, nrows=-1) -> pd.DataFrame:
    if nrows > 0:
        return pd.read_csv(path, nrows=nrows)
    else:
        return pd.read_csv(path)
