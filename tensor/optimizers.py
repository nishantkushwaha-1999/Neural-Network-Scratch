import numpy as np

class SGD():
    def __init__(self, params, lr) -> None:
        self.params = params
        self.lr = lr
    
    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad
    
    def __repr__(self) -> str:
        return f"SGD optimizer"