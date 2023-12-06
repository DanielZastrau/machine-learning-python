"""The code here is inspired by the cours, but was not taken from it.
I chose to reimplement the polynomial regression without the help of sklearn. 

Compare to the sklearn version a lot worse. The results are also vastly different.
"""

import numpy as np

np.random.seed(42)

import math

from typing import Union

from itertools import combinations_with_replacement

from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from icecream import ic

class LinearRegressor():
    """Find a best fit line for a given dataset to predict new data in the 1-D case.
    """

    def __init__(self, use_intercept: bool = True) -> None:
        self.a: np.ndarray = None
        self.b: np.ndarray = None
        
        self.use_intercept = use_intercept

    def _add_b(self, x: np.ndarray) -> np.ndarray:
        """Adjust the values slightly in order for the dimensions to be correct.
        """

        assert isinstance(x, np.ndarray), "x has to be a numpy array"

        intercepts = np.ones(shape=(x.shape[0]))

        return np.column_stack((intercepts, x))

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Computes the line ax + b. beta = (x.T * x)^(-1) * x.T * y
        """
        
        assert isinstance(x, np.ndarray), "x has to be a numpy array"
        assert isinstance(y, np.ndarray), "y has to be a numpy array"

        if self.use_intercept:
            x = self._add_b(x)

        inner = np.dot(x.T, x)

        # Permutes the matrix just a little bit to make it invertible
        while True:
            try:
                inv = np.linalg.inv(inner)
                break

            except Exception:
                max_elem_value = np.max(np.abs(inner))

                noise = np.random.normal(loc=0, scale=10**(-6)*max_elem_value, size=inner.shape)

                inner = inner + noise

        beta = np.dot(np.dot(inv, x.T), y)

        self.b = beta[0]
        self.a = beta[1:]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts the y value for a given value of x.
        """

        assert isinstance(x, np.ndarray), "x has to be a numpy array"

        return np.dot(x, self.a) + self.b

    def score(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        """
        
        assert isinstance(x, np.ndarray), "x has to be a numpy array"
        assert isinstance(y, np.ndarray), "y has to be a numpy array"

        y_pred = self.predict(x)

        return r2_score(y, y_pred)


class PolynomialFeatures():

    def __init__(self, degree: int) -> None:
        
        assert isinstance(degree, int), "degree has to be an int"

        self.degree = degree

    def _product(self, sample: np.ndarray, indeces: tuple) -> Union[int, float]:
        """Calculates the product of all elements from sample indexed by indeces.
        """

        if indeces == ():
            return 1

        out_num = 1
        for index in indeces:
            out_num *= sample[index]

        return out_num

    def _transformed_sample(self, sample: np.ndarray) -> np.ndarray:
        """Transform the sample according to self._output_features.
        """

        assert isinstance(sample, np.ndarray), "sample has to be a numpy array"

        return np.array([self._product(sample, indeces) for indeces in self._output_features ])

    def fit(self, x: np.ndarray, y: np.ndarray = None) -> None:
        """Computes how many output features there will be.

        y is an argument for api consistency.
        """
        
        assert isinstance(x, np.ndarray), "x has to be a numpy array"

        output_features = []
        for _degree_ in range(0, self.degree + 1):
            output_features.extend(list(combinations_with_replacement(range(0, x.shape[1]), _degree_)))

        self._output_features = output_features

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transforms the data to the amount of output features.
        """

        assert isinstance(x, np.ndarray), "x has to be a numpy array"

        return np.array([self._transformed_sample(sample) for sample in x])


def train_test_split(x: np.ndarray, y: np.ndarray, test_size: float) -> tuple:
    """Returns the data and target sets randomly splitted into train and test sets.
    """

    assert isinstance(x, np.ndarray), "x has to be a numpy array"
    assert isinstance(y, np.ndarray), "y has to be a numpy array"

    assert isinstance(test_size, float), "test_size has to be a float"
    assert 0 < test_size < 1, "test_size has to be between 0 and 1"

    num_samples = x.shape[0]
    num_test = math.ceil(num_samples * test_size)

    random_permutation = np.random.permutation(num_samples)
    test = random_permutation[ : num_test]
    train = random_permutation[num_test : ]

    x_train, x_test = x[train], x[test]
    y_train, y_test = y[train], y[test]

    return x_train, x_test, y_train, y_test


def test_poly_regression():
    """Test the polynomial regression implementation of sklearn.
    """

    dataset = fetch_california_housing()

    x = dataset.data[:, :]
    y = dataset.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    for degree in range(1, 6):

        pf = PolynomialFeatures(degree=degree)
        pf.fit(x_train, y_train)

        x_train_transformed = pf.transform(x_train)
        x_test_transformed = pf.transform(x_test)

        regr = LinearRegressor()
        regr.fit(x_train_transformed, y_train)

        r2 = regr.score(x_test_transformed, y_test)

        ic(degree, x_train.shape, x_train_transformed.shape, r2)

if __name__=="__main__":
    test_poly_regression()