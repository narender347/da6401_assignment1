import numpy as np
class SGD:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
    
    def step(self, weights, grads):
        for key in weights:
            grad_key = 'd' + key
            weights[key] -= self.lr * grads[grad_key]

class Momentum:
    def __init__(self, learning_rate=0.01, beta=0.9):
        self.lr = learning_rate
        self.beta = beta
        self.velocities = None
    
    def step(self, weights, grads):
        if self.velocities is None:
            self.velocities = {}
            for key in weights:
                self.velocities[key] = np.zeros_like(weights[key])
        
        for key in weights:
            grad_key = 'd' + key
            self.velocities[key] = self.beta * self.velocities[key] + (1 - self.beta) * grads[grad_key]
            weights[key] -= self.lr * self.velocities[key]

class Nesterov:
    def __init__(self, learning_rate=0.01, beta=0.9):
        self.lr = learning_rate
        self.beta = beta
        self.velocities = None
    
    def step(self, weights, grads):
        if self.velocities is None:
            self.velocities = {}
            for key in weights:
                self.velocities[key] = np.zeros_like(weights[key])
        
        for key in weights:
            grad_key = 'd' + key
            grad = grads[grad_key]
            self.velocities[key] = self.beta * self.velocities[key] + grad
            weights[key] -= self.lr * (self.beta * self.velocities[key] + grad)

class RMSprop:
    def __init__(self, learning_rate=0.01, beta=0.9, eps=1e-8):
        self.lr = learning_rate
        self.beta = beta
        self.eps = eps
        self.cache = None
    
    def step(self, weights, grads):
        if self.cache is None:
            self.cache = {}
            for key in weights:
                self.cache[key] = np.zeros_like(weights[key])
        
        for key in weights:
            grad_key = 'd' + key
            grad = grads[grad_key]
            self.cache[key] = self.beta * self.cache[key] + (1 - self.beta) * (grad ** 2)
            weights[key] -= self.lr * grad / (np.sqrt(self.cache[key]) + self.eps)

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0
    
    def step(self, weights, grads):
        if self.m is None:
            self.m = {}
            self.v = {}
            for key in weights:
                self.m[key] = np.zeros_like(weights[key])
                self.v[key] = np.zeros_like(weights[key])
        self.t += 1
        
        for key in weights:
            grad_key = 'd' + key
            grad = grads[grad_key]
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
            
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            weights[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class Nadam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0
    
    def step(self, weights, grads):
        if self.m is None:
            self.m = {}
            self.v = {}
            for key in weights:
                self.m[key] = np.zeros_like(weights[key])
                self.v[key] = np.zeros_like(weights[key])
        self.t += 1
        
        for key in weights:
            grad_key = 'd' + key
            grad = grads[grad_key]
            
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
            
            m_hat = (self.beta1 * self.m[key] / (1 - self.beta1 ** (self.t + 1))) + ((1 - self.beta1) * grad / (1 - self.beta1 ** self.t))
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            weights[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
