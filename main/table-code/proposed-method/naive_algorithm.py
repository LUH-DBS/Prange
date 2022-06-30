import pandas
import numpy


def find_unique_columns(table: pandas.DataFrame):
    uniques = []
    for columnname in table.columns:
        if column_is_unique(table[columnname]):
            uniques.append(table.columns.get_loc(columnname))
    return uniques


def column_is_unique(column: pandas.Series):
    def is_nan(x):
        return (x is numpy.nan or x != x)
    try:
        sorted_col = sorted(column.to_list(), key=lambda x:
                            float('-inf') if is_nan(x) else x)
    except TypeError as e:
        return False
    # use this code to recognize columns containing
    # a single NaN as not unique
    for i in range(1, len(sorted_col)):
        if is_nan(sorted_col[i-1]):
            return False
        if sorted_col[i-1] == sorted_col[i]:
            return False
    return True
