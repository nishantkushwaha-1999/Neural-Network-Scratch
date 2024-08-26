import numpy as np
from graphviz import Digraph
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

def visualize_model_decision_boundaries(model, X, y, c_colors: list, magnification=0.01, title="Title plot", ax=None,
                                        alpha=0.7):
    if len(y.shape) > 1:
        raise ValueError(f"y is supposed to be a vector")
    
    if len(X) != len(y):
        raise ValueError(f"X adn y are supposed to be of same length")
    
    n_classes = len(np.unique(y))
    if len(c_colors) != n_classes:
        raise ValueError(f"provide colors for all classes")
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, magnification),
                        np.arange(y_min, y_max, magnification))
    
    x_mesh = np.c_[xx.ravel(), yy.ravel()]

    proba_, _ = model.predict(x_mesh)

    labels = ['Class 0', 'Class 1', 'Class 2']
    added_labels = {0: False, 1: False, 2: False}

    for a, b, k in zip(X[:, 0], X[:, 1], y):
        if not added_labels[k]:
            ax.scatter(a, b, c=c_colors[k], label=labels[k], s=20)
            added_labels[k] = True
        else:
            ax.scatter(a, b, c=c_colors[k], s=20)

    
    for i in range(n_classes):
        ax.contourf(xx, 
                     yy, 
                     proba_[:, i].reshape(xx.shape),
                     cmap=LinearSegmentedColormap.from_list("", [ mcolors.to_rgba(c_colors[i], alpha=0), mcolors.to_rgba(c_colors[i], alpha=alpha) ]),
                     antialiased=True,
                     extend='min')
        


def trace(root):
    nodes, edges = set(), set()
    def build(root):
        if root not in nodes:
            nodes.add(root)
            for child in root._parent:
                edges.add((child, root))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    
    for n in nodes:
        dot.node(name=str(id(n)), label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot