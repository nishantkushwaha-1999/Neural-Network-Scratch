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
    
    def backprop(self):
        pass


class Relu:
    def forward(self, x):
        self.output = np.maximum(0, x)
        return self.output