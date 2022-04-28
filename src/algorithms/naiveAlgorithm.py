import pandas as pd

def find_unique_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Generate a list with all column ids which only contain unique values making use of sorting.

    Args:
        table (pd.Dataframe): the table to inspect

    Returns:
        pd.DataFrame: the indexes of the unique columns
    """
    tablelength = len(table)
    nunique = table.nunique().values
    unique_columns = []
    for index, value in enumerate(nunique):
        if value == tablelength:
            unique_columns.append(index)
    return unique_columns
