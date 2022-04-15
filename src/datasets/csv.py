from ._base import Baseclass
import pandas.io.sql as sqlio
import pandas


class CSV(Baseclass):

    def __init__(self, path) -> None:
        self.path = path

    def get_table(self, tableid: int, max_rows: int) -> pandas.DataFrame:
        """Return a complete table from the locally saved csv files with the id [tableid].

        Args:
            tableid (int): the id of the table
            max_rows (int): the maximum number of rows the returned table will have (only if flipped is False)

        Returns:
            list[list]: the table in a list format
        """
        if max_rows > 0:
            table = pandas.read_csv(f"{self.path}{tableid}.csv", nrows=max_rows)
        else:
            table = pandas.read_csv(f"{self.path}{tableid}.csv")
        return table
