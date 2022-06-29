import pandas as pd
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import recall

X = pd.read_csv('training_data.csv')
y = pd.read_csv('training_result.csv')

automl = AutoSklearnClassifier(
    time_left_for_this_task=train_time,
    metric=recall,
)
automl.fit(X, y)

return automl