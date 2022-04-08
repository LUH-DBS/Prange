from ._base import Baseclass
import numpy as np


class NaiveAlgorithmFlipped(Baseclass):

    def find_unique_columns(self, table: list[list], algorithm: str) -> list[int]:
        table = self._flip_table(table)
        return self.find_unique_columns_flipped(table, algorithm)

    def _flip_table(self, table: list[list]) -> list[list]:
        x = np.array(table)
        return x.T

    def find_unique_columns_flipped(self, table: list[list], algorithm: str) -> list[int]:
        """Generate a list with all column ids which only contain unique values making use of sorting.

        Args:
            table (list[list]): the table which has to be flipped (a list of columns, not of rows)
            algorithm (str): either 'hash' or 'sort', raises an error otherwise

        Raises:
            ValueError: if algorithm is neither 'hash' nor 'sort'

        Returns:
            list[int]: the indexes of the unique columns
        """

        def column_is_unique_sorting(column: list) -> bool:
            """Determine wether a column contains only unique values.

            Args:
                column (list): the column to test

            Returns:
                bool: True if unique, False otherwise
            """
            column = sorted(column)
            for index in range(len(column)):
                if index == 0:
                    continue
                if column[index] == column[index - 1]:  # TODO: what about NULL/NaN values?
                    return False
            return True

        def column_is_unique_hashing(column: list) -> bool:
            """Determine wether a column contains only unique values.

            Args:
                column (list): the column to test

            Returns:
                bool: True if unique, False otherwise
            """
            hashmap = {}
            for value in column:
                if value not in hashmap:
                    hashmap[value] = 0
                else:
                    return False
            return True

        match algorithm:
            case 'hash':
                column_is_unique = column_is_unique_hashing
            case 'sort':
                column_is_unique = column_is_unique_sorting
            case _:
                raise ValueError(
                    "Only 'hash' and 'sort' are valid algorithms.")

        uniques = []
        for index, column in enumerate(table):
            if column_is_unique(column):
                uniques.append(index)
        return uniques
