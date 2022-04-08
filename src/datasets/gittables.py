from ._base import Baseclass


class Gittables(Baseclass):

    def __init__(self, cursor) -> None:
        self.cursor = cursor

    def get_table(self, tableid: int, max_rows: int, flipped=False) -> list[list]:
        """Return a complete table from the gittables with the id [tableid].

        Args:
            tableid (int): the id of the table
            max_rows (int): the maximum number of rows the returned table will have (only if flipped is False)
            flipped (bool): if True, a list of columns will be returned, otherwise a list of rows

        Returns:
            list[list]: the table in a list format
        """
        table = []
        query = "SELECT num_rows, num_columns FROM gittables_tables_info WHERE id = %s"
        self.cursor.execute(query, (tableid,))
        row_count, column_count = self.cursor.fetchone()

        if flipped:
            query = "SELECT tokenized FROM gittables_main_tokenized WHERE tableid = %s and columnid = %s"
            for columnid in range(0, column_count):
                self.cursor.execute(query, (tableid, columnid))
                table.append([r[0] for r in self.cursor.fetchall()])
            return table
        else:
            if max_rows > 0:
                row_count = min(row_count, max_rows)
            query = "SELECT tokenized FROM gittables_main_tokenized WHERE tableid = %s and rowid = %s"
            for rowid in range(0, row_count):
                self.cursor.execute(query, (tableid, rowid))
                table.append([r[0] for r in self.cursor.fetchall()])
            return table

    def pretty_columns(self, tableid: int, columnids: list[int]) -> list[str | int | list]:
        """Return the column and table id together with its names.


        Args:
            tableid (int): th id of the table
            columnids (list[int]): a list with the column ids

        Returns:
            list[list]: a list in the format ['tableid', 'tablename', 'columnids', 'columnnames']
        """
        return [tableid, self.get_tablename(tableid), columnids, self.get_columnnames(tableid, columnids)]

    def pretty_columns_header(self) -> list[str]:
        return ['tableid', 'tablename', 'columnids', 'columnnames']

    def get_columnheader(self, tableid: int) -> list:
        """Get the column header for a table.

        Args:
            tableid (int): the id of the table

        Returns:
            list: the header ordered by column id
        """
        query = "SELECT header FROM gittables_columns_info WHERE tableid = %s ORDER BY columnid"
        self.cursor.execute(query, (tableid,))
        return [r[0] for r in self.cursor.fetchall()]

    def get_tablename(self, tableid: int) -> str:
        """Get the name of a table.

        Args:
            tableid (int): the id of the table

        Returns:
            str: the name of the table
        """
        query = "SELECT filename FROM gittables_tables_info WHERE id = %s"
        self.cursor.execute(query, (tableid,))
        return self.cursor.fetchone()[0]

    def get_columnnames(self, tableid: int, columnids: list[int]) -> list[str]:
        """Get the names of the columns

        Args:
            tableid (int): the id of the table
            columnids (list[int]): a list with the column ids

        Returns:
            list[str]: a list with the column names
        """
        names = self.get_columnheader(tableid)
        result = []
        for i in columnids:
            result.append(names[i])
        return result