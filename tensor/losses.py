import numpy as np

class MSE():
    def __init__(self) -> None:
        pass

    def __call__(self, y, y_pred):
        assert len(y) == len(y_pred)
        
        total_loss = sum((y-y_p)**2 for y, y_p in zip(y, y_pred)) / len(y)
        return total_loss[0]
    
    def __repr__(self) -> str:
        return f"MSE loss fn"

class Categorical_Cross_Entropy():
    def __init__(self) -> None:
        pass

    def __call__(self):
        print(f"CCE loss call")
    
    def __repr__(self):
        print(f"Categorical_Cross_Entropy loss fn")