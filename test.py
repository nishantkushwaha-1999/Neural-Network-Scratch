import network.layers as l
import numpy as np

x = np.array([[1,2,3,4],
              [1,1,1, 1]])

lyr1 = l.Dense(4, 1)
m = lyr1.forward(x)
act = l.Relu()
m = act.forward(m)