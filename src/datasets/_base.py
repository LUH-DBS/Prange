from abc import ABC


class Baseclass(ABC):
    def get_table(self, tableid: int, flipped: bool) -> list[list]:
        raise NotImplementedError()

    def pretty_columns(self, tableid: int, columnids: list[int]) -> list[str | int | list]:
        raise NotImplementedError()
