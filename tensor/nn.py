import random
import numpy as np
from tensor.tensor import Tensor

class Module():

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
    
    def parameters(self):
        params = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)

            if isinstance(attr, Module) and attr is not self:
                params += attr.parameters()
        return params
    
    def __call__(self, x):
        return self.forward(x)
    
    def __repr__(self) -> str:
        out = 'Model Short Summary::\n\n'
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Module) and attr is not self:
                out += f"{attr_name}: n_params: {len(attr.parameters())}\n"
        return out + f"\nTotal params: {len(self.parameters())}\n"



class Neuron(Module):
    def __init__(self, n_inputs, activation='relu', suffix='') -> None:
        super().__init__()
        self.w = np.array([Tensor(random.uniform(-1, 1), label=f"w-{suffix}-{_}") for _ in range(n_inputs)])
        self.b = Tensor(random.uniform(-1, 1), label=f"b-{suffix}")

        # For Testing
        # self.w = np.array([Tensor(1, label=f"w-{suffix}-{_}") for _ in range(n_inputs)])
        # self.b = Tensor(1, label=f"b-{suffix}")

        self.activation = activation

        self.n_inputs = n_inputs
    
    def __call__(self, x):
        if type(x) == np.ndarray:

            activations = np.matmul(x, self.w.T) + self.b
            
            if self.activation == 'lin':
                return activations
            elif self.activation == 'relu':
                return np.array([i.relu() for i in activations])
            elif self.activation == 'tanh':
                return np.array([i.tanh() for i in activations])
        
        else:
            raise TypeError(f"Input data must be a {np.ndarray}. Received {type(x)}")
    
    def parameters(self):
        return list(self.w) + [self.b]
    
    def __repr__(self) -> str:
        return f"Neuron(x={self.n_inputs})"


class layer_dense(Module):
    def __init__(self, n_neurons, n_inputs, activation='relu', suffix='') -> None:
        super().__init__()
        
        self.neurons = np.array([Neuron(n_inputs, activation, suffix=f"{suffix}-{_}") for _ in range(n_neurons)])
    
    def __call__(self, x):
        return np.array([n(np.array(x)) for n in self.neurons]).T
    
    def parameters(self):
        params = []
        for n in self.neurons:
            params += n.parameters()
        
        return params
    
    def __repr__(self) -> str:
        return f"dense_layer: {len(self.parameters())} params"

if __name__=="__main__":
    x = np.array([1, 2, 3])
    l = layer_dense(3, 3, 'relu')
    print(l(x))
    print(l)