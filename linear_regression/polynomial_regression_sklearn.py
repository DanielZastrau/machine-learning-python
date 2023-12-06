import numpy as np

np.random.seed(42)

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from icecream import ic

def test_sklearn_poly_regression():
    """Test the polynomial regression implementation of sklearn.
    """

    dataset = fetch_california_housing()

    x = dataset.data[:, :]
    y = dataset.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    for degree in range(1, 6):

        pf = PolynomialFeatures(degree=degree)
        pf.fit(x_train)

        x_train_transformed = pf.transform(x_train)
        x_test_transformed = pf.transform(x_test)

        regr = LinearRegression()
        regr.fit(x_train_transformed, y_train)

        r2 = regr.score(x_test_transformed, y_test)

        ic(degree, x_train.shape, x_train_transformed.shape, r2)

if __name__=="__main__":
    test_sklearn_poly_regression()