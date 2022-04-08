from abc import ABC


class Baseclass(ABC):
    def find_unique_columns(self, table: list[list], algorithm: str) -> list[int]:
        raise NotImplementedError()
