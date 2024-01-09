import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

from itertools import product

from icecream import ic

if __name__=="__main__":

    np.random.seed(42)

    dataset = load_iris()

    x: np.ndarray = dataset.data[:, :]
    y: np.ndarray = dataset.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    params = product([2], [500, 1_000, 1_500], [5, 10, 15])

    for n_clusters, max_iter, n_init in params:
        kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=n_init)
        kmeans.fit(x)

        y_pred = kmeans.predict(x_test)

        ic(n_clusters, max_iter, n_init, silhouette_score(x_test, y_pred))

    kmeans = KMeans(n_clusters=2, max_iter=500, n_init=5)