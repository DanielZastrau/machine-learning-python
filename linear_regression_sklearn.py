from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from icecream import ic


def test_sklearn_linear_regressor():
    """Currently I get the following error message, when running this.

    Traceback (most recent call last):
        File "C:\Users\z004ksam\Documents\Repos\machine-learning-python\linear_regression_sklearn.py", line 15, in <module>
            regr.fit(x_train, y_train)
        File "C:\Users\z004ksam\.virtualenvs\machine-learning-python--Gnu5ORe\lib\site-packages\sklearn\base.py", line 1145, in wrapper
            estimator._validate_params()
        AttributeError: 'numpy.ndarray' object has no attribute '_validate_params'
    """

    dataset = fetch_california_housing()

    x = dataset.data[:, :]
    y = dataset.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    regr = LinearRegression
    regr.fit(x_train, y_train)

    r2_score = regr.score(x_test, y_test)

    ic(r2_score)
    ic(regr.coef_)
    ic(regr.intercept_)
