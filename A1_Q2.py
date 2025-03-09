import numpy as np
from tensorflow.keras.datasets import fashion_mnist # type: ignore
import matplotlib.pyplot as plt

name_of_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = self.initialize_weights()
    
    def initialize_weights(self):
        np.random.seed(42)
        weights = {}
        for i in range(len(self.layers) - 1):
            weights[f'W{i+1}'] = np.random.randn(self.layers[i], self.layers[i+1]) * 0.1   #here we are initializing the weights randomly
            weights[f'b{i+1}'] = np.zeros((1, self.layers[i+1])) #here we are initializing the biases with zeros
        return weights
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)
    
    def forward_propagation(self, X):
        activations = {'A0': X}
        for i in range(len(self.layers) - 2):
            Z = np.dot(activations[f'A{i}'], self.weights[f'W{i+1}']) + self.weights[f'b{i+1}']
            activations[f'A{i+1}'] = self.relu(Z)
        Z_final = np.dot(activations[f'A{len(self.layers)-2}'], self.weights[f'W{len(self.layers)-1}']) + self.weights[f'b{len(self.layers)-1}']
        activations[f'A{len(self.layers)-1}'] = self.softmax(Z_final)
        return activations
    
    def predict(self, X):
        activations = self.forward_propagation(X)
        return activations[f'A{len(self.layers)-1}']  # Returning the probability distribution

def one_hot_encode(y, num_classes):
    encoded = np.zeros((y.shape[0], num_classes))
    encoded[np.arange(y.shape[0]), y] = 1
    return encoded

def load_data():
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
    X_train, X_test = X_train.reshape(-1, 784) / 255.0, X_test.reshape(-1, 784) / 255.0
    Y_train, Y_test = one_hot_encode(Y_train, 10), one_hot_encode(Y_test, 10)
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = load_data()
nn = NeuralNetwork([784, 128, 64, 10])

probabilities = nn.predict(X_test)
predictions = np.argmax(probabilities, axis=1)
accuracy = np.mean(np.argmax(Y_test, axis=1) == predictions)
print(f'Test Accuracy is : {accuracy * 100:.2f}%')


# Logging actual values vs predicted values
test_labels = np.argmax(Y_test, axis=1)
prediction_logs = []
for i in range(10):  # taking 10 sample predictions for logging
    predicted_class = predictions[i]
    true_class = test_labels[i]
    log_entry = {
        "Sample Index": i,
        "True Label": name_of_classes[true_class],
        "Predicted Label": name_of_classes[predicted_class],
        "Probabilities": [round(p, 4) for p in probabilities[i].tolist()]  # adjusted upto 4 decimal places for the predicted probabilities
    }
    prediction_logs.append(log_entry)
    print(f"Actual Class is : {Y_test[i].tolist()} {name_of_classes[true_class]}")
    print(f"The Predicted Class is: {[round(p, 4) for p in probabilities[i].tolist()]} {name_of_classes[predicted_class]}")

