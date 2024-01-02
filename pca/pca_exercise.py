import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

np.random.seed(42)

if __name__=="__main__":

    dataset = load_digits()

    x = dataset.data
    y = dataset.target

    print("Hello")