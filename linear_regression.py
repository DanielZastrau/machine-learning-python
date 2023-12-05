import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
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
        inv = np.linalg.inv(inner)
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

def test_simple_linear_regressor():
    dataset = fetch_california_housing()

    x: np.ndarray = dataset.data[:, :]
    y: np.ndarray = dataset.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    regr = LinearRegressor()
    regr.fit(x_train, y_train)
    r2 = regr.score(x_test, y_test)

    ic(r2)
    ic(regr.a)
    ic(regr.b)

if __name__=="__main__":
    test_simple_linear_regressor()