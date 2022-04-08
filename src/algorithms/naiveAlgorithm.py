from ._base import Baseclass


class NaiveAlgorithm(Baseclass):

    # Possible higher efficiency with the table as a numpy array instead of a list
    def find_unique_columns(self, table: list[list], algorithm: str) -> list[int]:
        """Generate a list with all column ids which only contain unique values making use of sorting.

        Args:
            table (list[list]): the table to inspect
            algorithm (str): either 'hash' or 'sort', raises an error otherwise

        Raises:
            ValueError: if algorithm is neither 'hash' nor 'sort'

        Returns:
            list[int]: the indexes of the unique columns
        """
        non_uniques = set([])
        hash_structure = [dict() for i in range(0, table[0].__len__())]
        for row in table:
            for index, value in enumerate(row):
                if value in hash_structure[index]:
                    non_uniques.add(index)
                else:
                    hash_structure[index][value] = 0
        return list(filter(lambda x: x not in non_uniques, range(0, table[0].__len__())))
