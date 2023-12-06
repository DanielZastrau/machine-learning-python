from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from numpy import std, var

from icecream import ic


def test_model_per_feature():
    """Test and compare the performance of the model trained on each feature alone.
    """

    dataset = fetch_california_housing()

    for index in range(dataset.data.shape[1]):
        x = dataset.data[:, [index]]
        y = dataset.target


        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

        regr = LinearRegression()
        regr.fit(x_train, y_train)


        r2_score = regr.score(x_test, y_test)

        ic("feature", index)
        ic(r2_score)
        ic(std(x))
        ic(var(x))
        ic("\n")

if __name__=="__main__":
    test_model_per_feature()