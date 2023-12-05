"""Exercise 1 of the udemy course 'Machine Learning Komplettkurs' by Jan Schaffranek
"""

import numpy as np

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

from itertools import combinations
from icecream import ic

# Set the seed, so that the scores are comparable across runs
np.random.seed(42)

def knn_ex() -> None:
    """Evaluates for which combination of hyperparameters the knn_classifier performs best.
    """

    # Load dataset
    iris = datasets.load_iris()

    # Extract relevant data
    data_features: np.ndarray = iris.data
    data_targets: np.ndarray = iris.target
    target_names: np.ndarray = iris.target_names

    # Asserts correct data types
    assert isinstance(data_features, np.ndarray), "wrong_data type"
    assert isinstance(data_targets, np.ndarray), "wrong_data type"
    assert isinstance(target_names, np.ndarray), "wrong_data type"

    # Get size for test and train sets
    num_samples: int = data_features.shape[0]
    num_features: int = data_features.shape[1]

    test_size = num_samples // 3

    # Randomize the samples for randomized test and train set selection
    randomized_sample_ids = np.random.permutation(num_samples)

    # Get all combinations of 2 features
    combs = combinations( range( len( data_features[0])), 2)

    # Evaluate model on a number of different hyperparameters
    for combination in combs:
        for weight in ["uniform", "distance"]:
            for neighbors in range(1, 10, 2):

                # Reduce the data features to a subset of 2 
                reduced_data_features = data_features[:, combination]
                
                # Select train and test sets
                ids_train, ids_test = randomized_sample_ids[:num_samples - test_size], randomized_sample_ids[num_samples - test_size:]
                samples_train, samples_test = reduced_data_features[ids_train, :], reduced_data_features[ids_test, :]
                targets_train, targets_test = data_targets[ids_train], data_targets[ids_test]

                # Initialize and fit the clf
                clf = KNeighborsClassifier(n_neighbors=neighbors, weights=weight)
                clf.fit(samples_train, targets_train)

                # Evaluate the clf
                print(f"features:{combination} -- weight:{weight} -- neighbors:{neighbors} --> accuracy = {clf.score(samples_test, targets_test)*100:.4}")

if __name__=="__main__":
    knn_ex()