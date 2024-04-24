import numpy as np
import matplotlib.pyplot as plt
import data_utils
import download

def download_data():
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    download_dir = "./data"
    download.maybe_download_and_extract(url, download_dir)

class KNearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, distance='L2'):
        if distance == 'L2':
            dists = self.compute_distances_L2(X)
        elif distance == 'L1':
            dists = self.compute_distances_L1(X)
        else:
            raise ValueError('Invalid distance metric')
        return self.predict_labels(dists, k=k)

    def compute_distances_L2(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.sqrt(np.sum(X**2, axis=1, keepdims=True) + np.sum(self.X_train**2, axis=1) - 2 * X.dot(self.X_train.T))
        return dists

    def compute_distances_L1(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i, :] = np.sum(np.abs(self.X_train - X[i, :]), axis=1)
        return dists

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = self.y_train[np.argsort(dists[i, :])[:k]]
            y_pred[i] = np.argmax(np.bincount(closest_y))
        return y_pred

def visualize_data(X_train, y_train):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()

def perform_cross_validation(classifier, X, y, k_choices, num_folds=5, distance='L2'):
    fold_size = X.shape[0] // num_folds
    X_folds = np.array_split(X, num_folds)
    y_folds = np.array_split(y, num_folds)
    k_to_accuracies = {}

    for k in k_choices:
        k_to_accuracies[k] = []
        for fold in range(num_folds):
            X_train = np.concatenate([X_folds[i] for i in range(num_folds) if i != fold])
            y_train = np.concatenate([y_folds[i] for i in range(num_folds) if i != fold])
            X_val = X_folds[fold]
            y_val = y_folds[fold]

            classifier.train(X_train, y_train)
            y_val_pred = classifier.predict(X_val, k=k, distance=distance)
            accuracy = np.mean(y_val_pred == y_val)
            k_to_accuracies[k].append(accuracy)
    
    return k_to_accuracies

def plot_cross_validation_results(k_choices, k_to_accuracies, title="Cross-validation Accuracy"):
    plt.figure(figsize=(12, 6))
    for k in k_choices:
        accuracies = k_to_accuracies[k]
        plt.scatter([k] * len(accuracies), accuracies)

    accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
    accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std, fmt='-o')
    plt.title(title)
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.xticks(k_choices)
    plt.show()


def compare_distance_metrics(classifier, X_train, y_train, X_test, y_test, k_choices):
    distances = ['L1', 'L2']
    distance_accuracies = {distance: [] for distance in distances}
    
    for k in k_choices:
        for distance in distances:
            classifier.train(X_train, y_train)
            y_test_pred = classifier.predict(X_test, k=k, distance=distance)
            accuracy = np.mean(y_test_pred == y_test)
            distance_accuracies[distance].append(accuracy)
            print(f"k = {k}, Distance metric = {distance}, Accuracy = {accuracy}")
    
    return distance_accuracies

def plot_distance_comparison(k_choices, distance_accuracies):
    plt.figure(figsize=(12, 6))
    for distance, accuracies in distance_accuracies.items():
        plt.plot(k_choices, accuracies, label=f"{distance} Distance", marker='o')
    
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('L1 vs L2 Distance Metric Accuracy')
    plt.xticks(k_choices)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    download_data()  
    cifar10_dir = 'data/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = data_utils.load_CIFAR10(cifar10_dir)
    visualize_data(X_train, y_train)
    # Reduce the data size for quicker execution.
    num_training = 5000
    mask = np.random.choice(range(50000), num_training, replace=False)
    X_train = X_train[mask].reshape(num_training, -1)
    y_train = y_train[mask]

    num_test = 500
    mask = np.random.choice(range(10000), num_test, replace=False)
    X_test = X_test[mask].reshape(num_test, -1)
    y_test = y_test[mask]

    classifier = KNearestNeighbor()
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
    
    # Perform 5-fold cross-validation using L2 distance and plot results.
    k_to_accuracies_L2 = perform_cross_validation(classifier, X_train, y_train, k_choices, num_folds=5, distance='L2')
    plot_cross_validation_results(k_choices, k_to_accuracies_L2, "Cross-validation accuracy for L2 distance")
    
    # Determine the best k from cross-validation for L2 distance.
    average_accuracies_L2 = {k: np.mean(v) for k, v in k_to_accuracies_L2.items()}
    best_k_L2 = max(average_accuracies_L2, key=average_accuracies_L2.get)
    print(f"Best k found by cross-validation using L2 distance: {best_k_L2}")

    
    # Directly compare L1 and L2 distance accuracies for each k value and plot comparison.
    distance_accuracies = compare_distance_metrics(classifier, X_train, y_train, X_test, y_test, k_choices)
    plot_distance_comparison(k_choices, distance_accuracies)
