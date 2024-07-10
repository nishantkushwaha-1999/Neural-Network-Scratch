import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

def visualize_model_decision_boundaries(model, X, y, c_colors: list, magnification=0.01, title="Title plot", ax=None):
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
                     cmap=LinearSegmentedColormap.from_list("", [ mcolors.to_rgba(c_colors[i], alpha=0), mcolors.to_rgba(c_colors[i], alpha=0.5) ]),
                     antialiased=True,
                     extend='min')