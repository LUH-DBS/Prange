from abc import ABC


class Baseclass(ABC):
    def get_table(self, tableid: int, flipped: bool) -> list[list]:
        raise NotImplementedError()

    def get_columninfo(self, tableid: int) -> list:
        raise NotImplementedError()

    def get_tablename(self, tableid: int) -> str:
        raise NotImplementedError()

    def get_columnnames(self, tableid: int, columnids: list[int]) -> list[str]:
        raise NotImplementedError()
