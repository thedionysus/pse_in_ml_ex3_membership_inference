import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def decision_tree_overfit(x, y):
    model = DecisionTreeClassifier()
    model = model.fit(x, y)
    return model