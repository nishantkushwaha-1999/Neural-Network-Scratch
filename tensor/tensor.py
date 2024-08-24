import math

class Tensor:
    def __init__(self, data, _children=(), _op='', label='') -> None:
        self.data = data
        self.grad = 0.0
        
        self._parent = set(_children)
        self._backward = lambda: None
        self._op = _op 
        self.label = label
    
    def exp(self):
        out = Tensor(math.exp(self.data), (self,), 'exp', label='exp')

        def _backward():
            self.grad += out.grad * out.data

        out._backward = _backward

        return out
    
    def ln(self):
        out = Tensor(math.log(self.data), (self,), 'ln', label='ln')

        def _backward():
            self.grad += out.grad * (1 / self.data)

        out._backward = _backward

        return out
    
    def relu(self):
        out = Tensor(max(0, self.data), (self,), 'reul', label='relu')

        def _backward():
            self.grad += out.grad * (1 if out.data > 0 else 0)

        out._backward = _backward
        
        return out
    
    def tanh(self):
        temp = 2 * self
        out = (temp.exp() - 1) / (temp.exp() + 1)
        return out

    def __repr__(self) -> str:
        return f"Tensor({self.data})"
    
    def __add__(self, right):
        right = right if isinstance(right, Tensor) else Tensor(right)
        out = Tensor(self.data + right.data, (self, right), '+', label='+')

        def _backward():
            self.grad += out.grad
            right.grad += out.grad
        
        out._backward = _backward

        return out
    
    def __mul__(self, right):
        right = right if isinstance(right, Tensor) else Tensor(right)
        out = Tensor(self.data * right.data, (self, right), '*', label='*')

        def _backward():
            self.grad += out.grad * right.data
            right.grad += out.grad * self.data
        
        out._backward = _backward

        return out
    
    def __pow__(self, n):
        out = Tensor(self.data ** n, (self,), f'**{n}', label=f'**{n}')

        def _backward():
            self.grad += out.grad * (n * (self.data ** (n-1)))

        out._backward = _backward

        return out

    def __neg__(self):
        return self * -1
    
    def __sub__(self, right):
        return self + (-right)
    
    def __radd__(self, right):
        return self + right
    
    def __rsub__(self, right):
        return (-self) + right

    def __rmul__(self, right):
        return self * right
    
    def __truediv__(self, right):
        return self * (right ** -1)
    
    def __rtruediv__(self, right):
        return (self**-1) * right

    def backward(self):
        self.grad = 1.0
        
        topo_map = []
        visited = set()
        def build_topo_map(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for parent in tensor._parent:
                    build_topo_map(parent)
                topo_map.append(tensor)
        
        build_topo_map(self)
        for i in range(len(topo_map) - 1, 0, -1):
            topo_map[i]._backward()