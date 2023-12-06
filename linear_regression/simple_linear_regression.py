import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from icecream import ic

class SimpleLinearRegressor():
    """Find a best fit line for a given dataset to predict new data in the 1-D case.
    """

    def __init__(self) -> None:
        self.a: int = None
        self.b: int = None

    def _comp_a(self, x: np.ndarray, y: np.ndarray) -> None:
        """Computes the value a, which is the slope of the line.
        """
        
        assert isinstance(x, np.ndarray), "x has to be a numpy array"
        assert isinstance(y, np.ndarray), "y has to be a numpy array"
        
        x_mean = np.mean(x, axis=0)
        y_mean = np.mean(y, axis=0)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean)**2)

        self.a = numerator / denominator

    def _comp_b(self, x: np.ndarray, y: np.ndarray) -> None:
        """Computes the value b, where the line intersects the y-axis.
        """
        
        assert isinstance(x, np.ndarray), "x has to be a numpy array"
        assert isinstance(y, np.ndarray), "y has to be a numpy array"

        x_mean = np.mean(x, axis=0)
        y_mean = np.mean(y, axis=0)

        self.b = y_mean - self.a * x_mean

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Computes the line ax + b.
        """
        
        assert isinstance(x, np.ndarray), "x has to be a numpy array"
        assert isinstance(y, np.ndarray), "y has to be a numpy array"

        self._comp_a(x, y)
        self._comp_b(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts the y value for a given value of x.
        """

        assert isinstance(x, np.ndarray), "x has to be a numpy array"

        return self.a.T * x + self.b

    def score(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        """
        
        assert isinstance(x, np.ndarray), "x has to be a numpy array"
        assert isinstance(y, np.ndarray), "y has to be a numpy array"

        y_pred = self.predict(x)
        y_mean = np.mean(y, axis=0)

        numerator = np.sum((y - y_pred)**2)
        denominator = np.sum((y - y_mean)**2)

        return 1.0 - (numerator / denominator)

def test_simple_linear_regressor():
    dataset = fetch_california_housing()

    x: np.ndarray = dataset.data[:, 0]
    y: np.ndarray = dataset.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    regr = SimpleLinearRegressor()
    regr.fit(x_train, y_train)
    r2 = regr.score(x_test, y_test)

    ic(r2)
    ic(regr.a)
    ic(regr.b)

if __name__=="__main__":
    test_simple_linear_regressor()