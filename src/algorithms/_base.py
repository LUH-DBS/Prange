from abc import ABC
import pandas

class Baseclass(ABC):
    def find_unique_columns(self, table: pandas.DataFrame) -> pandas.DataFrame:
        raise NotImplementedError()
