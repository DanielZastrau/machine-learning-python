import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

class KNearestNeighbors():

    def __init__(self, neighbors: int) -> None:
        self.neighbors = neighbors

    def fit(self, data_features: np.ndarray, data_classes: np.ndarray) -> None:

        self.data_features = data_features
        self.data_classes = data_classes

    def _distance(self, p1: np.ndarray, p2: np.ndarray) -> np.float64:
        """Euclidean norm equal to sqrt(sum(all components squared))
        """

        return np.linalg.norm(p1 - p2)

    def k_nearest_neighbors(self, new_data: np.ndarray) -> np.ndarray:
        """Find out the who the k nearest neighbors of a new datapoint are.
        """

        # standard list comprehension
        distances = np.array([[self._distance(point, datapoint) for datapoint in self.data_features] for point in new_data])

        # argsort sorts the elements and converts the elements to their id: [5, 3, 4] --> [1, 2, 0]
        # numpy specific indexing for multidimensional arrays, [:] applies argsort to each row individually
        ids_sorted_distances = np.argsort(distances[:])

        # numpy specific indexing for multidimensional arrays, [:, :self.neighbors] first ... elements of each row
        ids_k_nearest_neighbors = ids_sorted_distances[:, :self.neighbors]
        
        return ids_k_nearest_neighbors
        
    def vote_class(self, classes: np.ndarray) -> np.ndarray:
        """Decide class of a new datapoint by majority voting of nearest neighbors.
        """

        classes_, counts = np.unique(classes, return_counts=True)

        return classes_[np.argmax(counts)]


    def classify(self, new_data: np.ndarray) -> np.ndarray:
        """Classify new datapoints.
        """
        
        ids_k_nearest_neighbors = self.k_nearest_neighbors(new_data)

        return np.array([self.vote_class(self.data_classes[ids]) for ids in ids_k_nearest_neighbors])
    
def generate_data_set() -> tuple:
        
        num_class_one, mean_class_one = 10, np.array([0, 0])
        num_class_two, mean_class_two = 6, np.array([-10, 4])
        num_class_three, mean_class_three = 13, np.array([10, 10])

        cov = np.array([[1, 0], [0, 1]])

        data_one = np.random.multivariate_normal(mean_class_one, cov, num_class_one)
        data_two = np.random.multivariate_normal(mean_class_two, cov, num_class_two)
        data_three = np.random.multivariate_normal(mean_class_three, cov, num_class_three)

        data = np.concatenate((data_one, data_two, data_three), axis=0)
        classes = np.array([0 for _ in range(num_class_one)] \
                                + [1 for _ in range(num_class_two)] \
                                + [2 for _ in range(num_class_three)])

        return data, classes

def generate_new_data(how_much: int) -> np.ndarray:

    return np.array([np.random.uniform(low=-20, high=20, size=2) for _ in range(how_much)])

def classify_new_data(data_features: np.ndarray, data_classes: np.ndarray, new_data_features: np.ndarray) -> np.ndarray:

    clf = KNearestNeighbors(5)
    clf.fit(data_features, data_classes)

    new_data_classes = clf.classify(new_data_features)

    return new_data_classes

def show_data_set(data_features: np.ndarray, data_classes: np.ndarray,
                    new_data_features: np.ndarray, new_data_classes: np.ndarray) -> None:
    
    colors = ["red", "blue", "green"]

    for datapoint, dataclass in zip(data_features, data_classes):
        plt.scatter(datapoint[0], datapoint[1], color=colors[dataclass])

    for datapoint, dataclass in zip(new_data_features, new_data_classes):
        plt.scatter(datapoint[0], datapoint[1], color=colors[dataclass], marker="x", s=200)

    plt.show()


if __name__ == "__main__":

    data_features, data_classes = generate_data_set()
    new_data_features = generate_new_data(how_much=40)
    new_data_classes = classify_new_data(data_features, data_classes, new_data_features)
    show_data_set(data_features, data_classes, new_data_features, new_data_classes)