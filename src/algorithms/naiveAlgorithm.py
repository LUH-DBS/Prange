from ._base import Baseclass
import pandas as pd


class NaiveAlgorithm(Baseclass):

    # Possible higher efficiency with the table as a numpy array instead of a list
    def find_unique_columns(self, table: pd.DataFrame) -> list:
        """Generate a list with all column ids which only contain unique values making use of sorting.

        Args:
            table (list[list]): the table to inspect
            algorithm (str): either 'hash' or 'sort', raises an error otherwise

        Raises:
            ValueError: if algorithm is neither 'hash' nor 'sort'

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
