from ._base import Baseclass
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype


class MachineLearning(Baseclass):
    header = ["Duplicates", "Data Type", "Sorted",
              # number
              "Min. value", "Max. value", "Mean", "Std. Deviation",
              # string
              "Avg. string length", "Min. string length", "Max. string length"
              # date?
              ]  # 10

    # Possible higher efficiency with the table as a numpy array instead of a list
    def find_unique_columns(self, table: pd.DataFrame) -> pd.DataFrame:
        """Generate a list with all column ids which only contain unique values making use of machine learning.

        Args:
            table (pd.Dataframe): the table to inspect

        Returns:
            pd.DataFrame: the indexes of the unique columns
        """
        print(self.prepare_table(table))

    def prepare_table(self, table: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(columns=self.header)
        for column_id in table:
            result = pd.concat([result, self.prepare_column(
                table[column_id])])
        return result

    def prepare_column(self, column: pd.DataFrame) -> pd.DataFrame:
        # return immediatly if there are any duplicated
        if column.duplicated().any():
            # 10
            return pd.DataFrame([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], columns=self.header, index=[column.name])
        # duplicate = 0, data type and sorted will be changed
        result = [0, 0, 0]
        # check if entries are sorted
        try:
            if all(column[i+1] >= column[i] for i in range(0, len(column)-1)):
                result[2] = 1
            if all(column[i+1] <= column[i] for i in range(0, len(column)-1)):
                result[2] = 1
        except:
            print("Error: column " + column.name)
        # handle integer and float
        if is_numeric_dtype(column):
            result[1] = 1
            description = column.describe()
            result.append(description['min'])
            result.append(description['max'])
            result.append(description['mean'])
            result.append(description['std'])
            # values for strings
            result += [0, 0, 0]
            return pd.DataFrame([result], columns=self.header, index=[column.name])
        if is_string_dtype(column):
            result[1] = 2
            # values for numbers
            result += [0, 0, 0, 0]
            try:
                result += self._describe_string(column)
            except:
                result[1] = 3  # mixed column
                result += [0, 0, 0]
            return pd.DataFrame([result], columns=self.header, index=[column.name])
        raise NotImplementedError("Not implemented column type")

    def _describe_string(self, column: pd.DataFrame) -> list:
        # "Avg. string length", "Min. string length", "Max. string length"
        length_list = []
        for value in column.values:
            if isinstance(value, str):
                length_list.append(len(value))
            else:
                raise ValueError("Not a String")
        average = sum(length_list)/len(length_list)
        minimum = min(length_list)
        maximum = max(length_list)
        return [average, minimum, maximum]
