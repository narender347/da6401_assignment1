import numpy as np
from tensorflow.keras.datasets import fashion_mnist # type: ignore
from optimizer import * #refer optimizer.py file for different classes of gradient descent

name_of_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class NeuralNetwork:   #creating a NeuralNetwork class which can be accessed easily from anyother file
    def __init__(self, no_of_layers):
        self.no_of_layers = no_of_layers
        self.weights = self.initialize_weights()
    
    def initialize_weights(self):  #randomly initializing the weights and biases 
        np.random.seed(42)
        weights = {}
        for i in range(len(self.no_of_layers) - 1):
            weights[f'W{i+1}'] = np.random.randn(self.no_of_layers[i], self.no_of_layers[i+1]) * 0.1
            weights[f'b{i+1}'] = np.zeros((1, self.no_of_layers[i+1]))
        return weights
    
    def relu(self, Z):    #relu activation function
        return np.maximum(0, Z)
    
    def softmax(self, Z):     #softmax activation function
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)
    
    def forward_propagation(self, X): #code for forward propagation
        activation_fn = {'A0': X}
        for i in range(len(self.no_of_layers) - 2):
            Z = np.dot(activation_fn[f'A{i}'], self.weights[f'W{i+1}']) + self.weights[f'b{i+1}']
            activation_fn[f'A{i+1}'] = self.relu(Z)
        Z_final = np.dot(activation_fn[f'A{len(self.no_of_layers)-2}'], self.weights[f'W{len(self.no_of_layers)-1}']) + self.weights[f'b{len(self.no_of_layers)-1}']
        activation_fn[f'A{len(self.no_of_layers)-1}'] = self.softmax(Z_final)
        return activation_fn
    
    def calculate_loss(self, Y, A_last):   #computing the loss 
        m = Y.shape[0]
        return -np.sum(Y * np.log(A_last + 1e-8)) / m
    
    def back_propagation(self, activation_fn, Y):   #backward propagation to minimise the loss
        grads = {}
        m = Y.shape[0]
        L = len(self.no_of_layers) - 1
        
        dZ = activation_fn[f'A{L}'] - Y
        grads[f'dW{L}'] = np.dot(activation_fn[f'A{L-1}'].T, dZ) / m     
        grads[f'db{L}'] = np.sum(dZ, axis=0, keepdims=True) / m
        
        for l in reversed(range(1, L)):
            dA_prev = np.dot(dZ, self.weights[f'W{l+1}'].T)
            dZ = dA_prev * (activation_fn[f'A{l}'] > 0)
            grads[f'dW{l}'] = np.dot(activation_fn[f'A{l-1}'].T, dZ) / m
            grads[f'db{l}'] = np.sum(dZ, axis=0, keepdims=True) / m
        
        return grads
    
    def fit(self, X_train, Y_train, optimizer, no_of_epochs, batch_size, X_val=None, Y_val=None):
        num_samples = X_train.shape[0]
        for epoch in range(no_of_epochs):
            permutation = np.random.permutation(num_samples)
            X_shuffled = X_train[permutation]
            Y_shuffled = Y_train[permutation]
            
            for i in range(0, num_samples, batch_size):  #considers the given batch size accordingly
                X_batch = X_shuffled[i:i+batch_size]
                Y_batch = Y_shuffled[i:i+batch_size]
                
                activation_fn = self.forward_propagation(X_batch)
                A_last = activation_fn[f'A{len(self.no_of_layers)-1}']
                loss = self.calculate_loss(Y_batch, A_last)
                
                grads = self.back_propagation(activation_fn, Y_batch)
                optimizer.step(self.weights, grads)
            
            if X_val is not None and Y_val is not None:
                val_activation_fn = self.forward_propagation(X_val)
                val_loss = self.calculate_loss(Y_val, val_activation_fn[f'A{len(self.no_of_layers)-1}'])
                val_pred = np.argmax(val_activation_fn[f'A{len(self.no_of_layers)-1}'], axis=1)
                val_accuracy = np.mean(np.argmax(Y_val, axis=1) == val_pred)
                print(f'Epoch {epoch+1}/{no_of_epochs}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f},  Accuracy: {val_accuracy*100:.2f} %')
            else:
                print(f'Epoch {epoch+1}/{no_of_epochs}, Loss: {loss:.4f}')
    
    def predict(self, X):
        activation_fn = self.forward_propagation(X)
        return np.argmax(activation_fn[f'A{len(self.no_of_layers)-1}'], axis=1)
    
    def evaluate(self, X, Y):  #returns the values of the loss and accuracy
        activation_fn = self.forward_propagation(X)
        A_last = activation_fn[f'A{len(self.no_of_layers)-1}']
        loss = self.calculate_loss(Y, A_last)
        pred = np.argmax(A_last, axis=1)
        accuracy = np.mean(np.argmax(Y, axis=1) == pred)
        return loss, accuracy


def one_hot_encode(y, NO_classes): #one indicates the presence of class
    encoded = np.zeros((y.shape[0], NO_classes))
    encoded[np.arange(y.shape[0]), y] = 1
    return encoded


def load_data():
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
    X_train = X_train.reshape(-1, 784) / 255.0
    X_test = X_test.reshape(-1, 784) / 255.0
    Y_train = one_hot_encode(Y_train, 10)
    Y_test = one_hot_encode(Y_test, 10)
    return X_train, Y_train, X_test, Y_test

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_data()
    
    no_of_layers = [784, 128, 64, 10]
    model = NeuralNetwork(no_of_layers)
    
    optimizer = Nadam(learning_rate=0.01)
    
    model.fit(X_train, Y_train, optimizer, no_of_epochs=10, batch_size=256, X_val=X_test, Y_val=Y_test)
    
    test_loss, test_accuracy = model.evaluate(X_test, Y_test)
    print(f'Test Accuracy: {test_accuracy*100:.2f}%')
