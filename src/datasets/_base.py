from abc import ABC


class Baseclass(ABC):
    def get_table(self, tableid: int, max_rows) -> list[list]:
        raise NotImplementedError()

    def pretty_columns(self, tableid: int, columnids: list[int]) -> list[str | int | list]:
        raise NotImplementedError()

    def pretty_columns_header(self) -> list[str]:
        raise NotImplementedError()

    def pathname(self) -> str:
        raise NotImplementedError()
