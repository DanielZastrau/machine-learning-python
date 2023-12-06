"""Applies the regressor to an adjusted cos function and plots the best fit line against it.
"""

import numpy as np
import matplotlib.pyplot as plt

from polynomial_regression import PolynomialFeatures
from polynomial_regression import LinearRegressor

def f(x: np.ndarray) -> np.ndarray:
    return -(x**4) * np.cos(x)

x = np.arange(start=0.0, stop=10.0, step=0.2).reshape(-1, 1)
y = f(x)

colors = ["blue", "red", "green"]

def plot_poly_reg(x: np.ndarray, y: np.ndarray, degree: int) -> None:
    # Preprocessing
    pf = PolynomialFeatures(degree=degree)
    pf.fit(x)

    x_transformed = pf.transform(x)

    poly_regr = LinearRegressor()
    poly_regr.fit(x_transformed, y)
    r2_score = poly_regr.score(x_transformed, y)

    print(f"Score: {r2_score} for degree: {degree}")
    print(f"Coef: {poly_regr.a}")
    print(f"Intercept: {poly_regr.b}")

    y_pred = poly_regr.predict(x_transformed)

    # Plotting
    _ = plt.figure(figsize=(8, 8))
    plt.plot(x, y, color="lightblue", linewidth=2, label="GT")
    plt.scatter(x, y, color="white", s=10, marker="o", label="Dataset")
    plt.plot(
        x,
        y_pred,
        color=colors[degree - 1],
        linewidth=2,
        label=f"Degree {degree}",
    )
    plt.show()

for degree in [1, 2, 3]:
    plot_poly_reg(x, y, degree)