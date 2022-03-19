from ._base import Baseclass


class Maintables(Baseclass):

    def __init__(self, cursor) -> None:
        self.cursor = cursor

    def get_table(self, tableid: int, flipped: bool) -> list[list]:
        """Return a complete table from the maintable with the id [tableid].

        Args:
            tableid (int): the id of the table
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
            # table.append(self.get_columninfo(self.cursor, tableid))
            query = "SELECT tokenized FROM main_tokenized WHERE tableid = %s and rowid = %s"
            for rowid in range(0, row_count):
                self.cursor.execute(query, (tableid, rowid))
                table.append([r[0] for r in self.cursor.fetchall()])
            return table

    def get_columninfo(self, tableid: int) -> list:
        """Get the column header for a table.

        Args:
            tableid (int): the id of the table

        Returns:
            list: the header
        """
        query = "SELECT header FROM gittables_columns_info WHERE tableid = %s"
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
        names = self.get_columninfo(tableid)
        result = []
        for i in columnids:
            result.append(names[i])
        return result
