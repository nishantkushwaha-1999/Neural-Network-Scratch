import numpy as np

class Dense:
    def __init__(self, input_size, neurons):
        self.weights = np.random.randn(input_size, neurons) / 10
        self.bias = np.zeros((1, neurons))
    
    def forward(self, x):
        if x.shape[-1] != self.weights.shape[0]:
            raise ValueError(f"input error")
        
        self.output = np.dot(x, self.weights) + self.bias
        return self.output
    
    def backward(self, derivatives):
        self.prev_devrivatives = derivatives.deepcopy()


class Relu:
    def forward(self, x):
        self.output = np.maximum(0, x)
        return self.output
    
    def backward(self, derivatives):
        self.prev_devrivatives = derivatives.deepcopy()


class  Softmax:
    def forward(self, x):
        epowerx = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = epowerx / np.sum(epowerx, axis=1, keepdims=True)
        return self.output
    
    def backward(self, derivatives):
        self.prev_devrivatives = derivatives.deepcopy()