from ._base import Baseclass


class Maintables(Baseclass):

    def __init__(self, cursor) -> None:
        self.cursor = cursor

    def get_table(self, tableid: int, max_rows: int, flipped=False) -> list[list]:
        """Return a complete table from the maintables with the id [tableid].

        Args:
            tableid (int): the id of the table
            max_rows (int): the maximum number of rows the returned table will have (only if flipped is False)
            flipped (bool): if True, a list of columns will be returned, otherwise a list of rows

        Returns:
            list[list]: the table in a list format
        """
        table = []
        query = "SELECT COUNT(DISTINCT colid), COUNT(DISTINCT rowid) FROM main_tokenized WHERE tableid = %s"
        self.cursor.execute(query, (tableid,))
        row_count, column_count = self.cursor.fetchone()

        if flipped:
            query = "SELECT tokenized FROM main_tokenized WHERE tableid = %s and colid = %s"
            for columnid in range(0, column_count):
                self.cursor.execute(query, (tableid, columnid))
                table.append([r[0] for r in self.cursor.fetchall()])
            return table
        else:
            if max_rows > 0:
                row_count = min(row_count, max_rows)
            query = "SELECT tokenized FROM main_tokenized WHERE tableid = %s and rowid = %s"
            for rowid in range(0, row_count):
                self.cursor.execute(query, (tableid, rowid))
                table.append([r[0] for r in self.cursor.fetchall()])
            return table

    def pretty_columns(self, tableid: int, columnids: list[int]) -> list[str | int | list]:
        """Return the column together with the table id.


        Args:
            tableid (int): th id of the table
            columnids (list[int]): a list with the column ids

        Returns:
            list[list]: a list in the format ['tableid', 'columnids', ]
        """
        return [tableid, columnids]

    def pretty_columns_header(self) -> list[str]:
        return ['tableid', 'columnids', ]