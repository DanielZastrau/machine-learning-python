from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from icecream import ic


def test_sklearn_linear_regressor():
    """Test the performance of the sklearn Linear Regressor on the complete CaliHousing dataset.
    """

    dataset = fetch_california_housing()

    x = dataset.data[:, :]
    y = dataset.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    regr = LinearRegression()
    regr.fit(x_train, y_train)

    r2_score = regr.score(x_test, y_test)

    ic(r2_score)
    ic(regr.coef_)
    ic(regr.intercept_)

if __name__=="__main__":
    test_sklearn_linear_regressor()