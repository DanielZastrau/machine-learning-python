import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


if __name__=="__main__":

    np.random.seed(42)

    dataset = load_digits()

    x = dataset.data.astype(np.float32)
    y = dataset.target.astype(np.float32)

    # Without pca
    print("WITHOUT PCA", "-"*100)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    clf = SVC(C=1.5, degree=2, kernel="poly", max_iter=100)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print(clf.score(x_test, y_test))
    print(confusion_matrix(y_true=y_test, y_pred=y_pred))


    # With pca
    print("WITH PCA", "-"*100)
    print(x.shape)
    pca = PCA(n_components=0.9, svd_solver="full", copy=True)
    pca.fit(x)
    x_pca = pca.transform(x)

    print(sum(pca.explained_variance_ratio_))
    print(x_pca.shape)

    x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.33)

    clf = SVC(C=1, degree=2, kernel="rbf", max_iter=-1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print(clf.score(x_test, y_test))
    print(confusion_matrix(y_test, y_pred))

