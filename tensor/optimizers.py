import numpy as np

class SGD():
    def __init__(self, model, lr=0.01):
        self.params = model.parameters()
        self.lr = lr
    
    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad
    
    def __repr__(self) -> str:
        return f"SGD optimizer"