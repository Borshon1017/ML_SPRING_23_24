import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Generate synthetic dataset
X, Y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_classes=5, n_clusters_per_class=1, random_state=42)

# Convert labels to one-hot encoding
Y_one_hot = np.eye(5)[Y]

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_one_hot, test_size=0.2, random_state=42)

class MultiLayerNN(object):
    def __init__(self):
        # Network architecture
        self.input_neurons = 20
        self.hidden_neurons = [64, 64, 64]  # Three hidden layers
        self.output_neurons = 5  # Five classes

        # Learning rate
        self.learning_rate = 0.01

        # Weights initialization
        self.W1 = np.random.randn(self.input_neurons, self.hidden_neurons[0])
        self.W2 = np.random.randn(self.hidden_neurons[0], self.hidden_neurons[1])
        self.W3 = np.random.randn(self.hidden_neurons[1], self.hidden_neurons[2])
        self.W4 = np.random.randn(self.hidden_neurons[2], self.output_neurons)

    def sigmoid(self, x, derivative=False):
        if derivative:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def cross_entropy(self, Y, Y_pred):
        m = Y.shape[0]
        loss = -np.sum(Y * np.log(Y_pred + 1e-12)) / m
        return loss

    def feedForward(self, X):
        # Forward pass
        Z1 = np.dot(X, self.W1)
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(A1, self.W2)
        A2 = self.sigmoid(Z2)
        Z3 = np.dot(A2, self.W3)
        A3 = self.sigmoid(Z3)
        Z4 = np.dot(A3, self.W4)
        A4 = self.softmax(Z4)
        return A1, A2, A3, A4

    def backPropagation(self, X, Y, A1, A2, A3, A4):
        # Backward pass
        m = Y.shape[0]
        dZ4 = A4 - Y
        dW4 = np.dot(A3.T, dZ4) / m
        dZ3 = np.dot(dZ4, self.W4.T) * self.sigmoid(A3, derivative=True)
        dW3 = np.dot(A2.T, dZ3) / m
        dZ2 = np.dot(dZ3, self.W3.T) * self.sigmoid(A2, derivative=True)
        dW2 = np.dot(A1.T, dZ2) / m
        dZ1 = np.dot(dZ2, self.W2.T) * self.sigmoid(A1, derivative=True)
        dW1 = np.dot(X.T, dZ1) / m

        # Update weights
        self.W1 -= self.learning_rate * dW1
        self.W2 -= self.learning_rate * dW2
        self.W3 -= self.learning_rate * dW3
        self.W4 -= self.learning_rate * dW4
        
    def train(self, X, Y, epochs=1000):
        loss_history = []
        for epoch in range(epochs):
            A1, A2, A3, A4 = self.feedForward(X)
            self.backPropagation(X, Y, A1, A2, A3, A4)
            loss = self.cross_entropy(Y, A4)
            loss_history.append(loss)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
        return loss_history

# Initialize the network
nn = MultiLayerNN()

# Train the network and store the loss history
loss_history = nn.train(X_train, Y_train, epochs=1000)

# Plot the training loss
plt.plot(loss_history)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Cross-entropy Loss')
plt.show()

# Predict on test set
_, _, _, Y_pred = nn.feedForward(X_test)
Y_pred_labels = np.argmax(Y_pred, axis=1)
Y_test_labels = np.argmax(Y_test, axis=1)

# Calculate accuracy
accuracy = accuracy_score(Y_test_labels, Y_pred_labels)
print(f"Accuracy: {accuracy}")

# Detailed performance analysis
print(classification_report(Y_test_labels, Y_pred_labels))
