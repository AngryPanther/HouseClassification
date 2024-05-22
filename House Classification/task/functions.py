import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def dtc():
    return DecisionTreeClassifier(criterion='entropy',
                                  max_features=3,
                                  splitter='best',
                                  max_depth=6,
                                  min_samples_split=4,
                                 random_state=3)
