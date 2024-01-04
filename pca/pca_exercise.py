import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 

np.random.seed(42)

if __name__=="__main__":

    dataset = load_digits()

    x = dataset.data
    y = dataset.target

    print(x.shape)

    n_components = 21
    pca = PCA(n_components=n_components, copy=True)
    pca.fit(x)
    x_pca = pca.transform(x)

    # apply knn to transformed data
    x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.3)

    clf = KNeighborsClassifier(n_neighbors=3, weights="uniform")
    clf.fit(x_train, y_train)

    # apply knn to non-transformed data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    clf = KNeighborsClassifier(n_neighbors=7, weights="distance")
    clf.fit(x_train, y_train)