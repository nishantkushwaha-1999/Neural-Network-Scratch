import numpy as np
import copy

class Dense:
    def __init__(self, input_size, neurons):
        self.weights = np.random.randn(input_size, neurons) / 10
        self.bias = np.zeros((1, neurons))
    
    def forward(self, x):
        self.x = x.copy()
        if x.shape[-1] != self.weights.shape[0]:
            raise ValueError(f"input error")
        
        self.output = np.dot(x, self.weights) + self.bias
        return self.output
    
    def derivative_self(self, derivatives):
        d_weights = np.dot(self.x.T, derivatives)
        d_inputs = np.dot(derivatives, self.weights.T)
        d_biases = np.sum(derivatives, axis=0, keepdims=True)
        return d_weights, d_inputs, d_biases

    def backward(self, derivatives):
        prev_derivatives = derivatives.copy()
        self.d_weights, self.d_inputs, self.d_biases = self.derivative_self(prev_derivatives)
        return self.d_weights, self.d_inputs, self.d_biases


class Relu:
    def forward(self, x):
        self.x = x.copy()
        self.output = np.maximum(0, x)
        return self.output
    
    def derivative_self(self):
        d_inputs = copy.deepcopy(self.x > 0)
        d_inputs = d_inputs.astype(int)
        return d_inputs

    def backward(self, derivatives):
        derivatives = derivatives.copy()
        d_inputs = self.derivative_self()
        derivatives[~d_inputs.astype(bool)] = 0
        self.d_inputs = derivatives
        return self.d_inputs


class  Softmax:
    def forward(self, x):
        self.x = x.copy()
        epowerx = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = epowerx / np.sum(epowerx, axis=1, keepdims=True)
        return self.output
    
    def derivative_self(self, derivatives):
        d_inputs = np.empty_like(derivatives)
        for index, (output, derivative) in enumerate(zip(self.output, derivatives)):
            # print('out', output)
            output = output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(output) - np.dot(output, output.T)
            # print('jaqc', jacobian_matrix, 'out', output)
            d_inputs[index] = np.dot(jacobian_matrix, derivative)
        
        return d_inputs
    
    def backward(self, derivatives):
        self.d_inputs = derivatives.copy()
        self.d_inputs = self.derivative_self(self.d_inputs)
        return self.d_inputs