from ._base import Baseclass
import pandas as pd


class MachineLearning(Baseclass):

    # Possible higher efficiency with the table as a numpy array instead of a list
    def find_unique_columns(self, table: pd.DataFrame) -> pd.DataFrame:
        """Generate a list with all column ids which only contain unique values making use of machine learning.

        Args:
            table (pd.Dataframe): the table to inspect

        Returns:
            pd.DataFrame: the indexes of the unique columns
        """
        pass

    def prepare_table(self, table: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame()
        for column in table:
            result = pd.concat(result, self.prepare_column(column))
        return result

    def prepare_column(self, column: pd.DataFrame)-> pd.DataFrame:
        pass
