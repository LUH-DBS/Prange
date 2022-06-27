if has_duplicates(column):
  return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
result = [0, 0, 0]
# check if entries are sorted
try:
  if all(column[i+1] >= column[i] for i in range(len(column)-1)):
      result[2] = 1
  if all(column[i+1] <= column[i] for i in range(len(column)-1)):
      result[2] = 1
except TypeError:
  # mostly this means the column contains None/NaN values
  pass
if only_bool(column):
  result[1] = 3
  result += [0, 0, 0, 0, 0, 0, 0]
  return result
if only_numeric(column):
  result[1] = 1
  result += [min_value(column), max_value(column),
             mean_value(column), std_deviation(column)]
  # values for strings
  result += [0, 0, 0]
  return result
# values for numbers
result += [0, 0, 0, 0]
try:
    length_list = []
    for value in column:
        if isinstance(value, str):
            length_list.append(len(value))
        else:
            raise ValueError("Not a String")
    if len(length_list) == 0:
        average = 0
    else:
        average = sum(length_list)/len(length_list)
    minimum = min(length_list)
    maximum = max(length_list)
    result[1] = 2
    result += [average, minimum, maximum]
except ValueError:
    result[1] = 4  # mixed column, mostly None/NaN value
    result += [0, 0, 0]
return result
