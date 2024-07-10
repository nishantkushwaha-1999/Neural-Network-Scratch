import numpy as np

class SGD:
    def __init__(self, lr=1.0) -> None:
        self.learning_rate = lr
    
    def update_weights(self, layer):
        layer.weights -= self.learning_rate * layer.d_weights
        layer.bias -= self.learning_rate * layer.d_biases