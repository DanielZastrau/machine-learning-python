import numpy as np
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

np.random.seed(42)

if __name__=="__main__":

    dataset = load_wine()
    x = dataset.data
    y = dataset.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    parameters = {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 2, 4, 6, 8],
        "min_samples_leaf": [2, 4],
        "max_features": ["sqrt", "log2"]
    }

    clf = DecisionTreeClassifier()

    grid_cv = GridSearchCV(clf, parameters, cv=3)
    _ = grid_cv.fit(x_train, y_train)

    print(f"Best parameters set found on development set: {grid_cv.best_params_}\n")

    means = grid_cv.cv_results_["mean_test_score"]
    stds = grid_cv.cv_results_["std_test_score"]

    for mean, std, params in zip(means, stds, grid_cv.cv_results_["params"]):
        print(f"{mean:.3f} (+/-{2*std:.3f}) for {params}")

    clf = DecisionTreeClassifier(**grid_cv.best_params_)
    _ = clf.fit(x_train, y_train)

    acc = clf.score(x_test, y_test)
    print(f"Test Acc: {acc}")