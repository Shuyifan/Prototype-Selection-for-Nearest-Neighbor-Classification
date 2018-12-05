from sklearn.datasets import load_digits
from scipy.spatial import distance_matrix
import numpy as np
from classifier import classifier
import matplotlib.pyplot as plt

digits = load_digits()

X = digits.data
y = digits.target

D = distance_matrix(X, X)

obj_vals = list()
for i in range(2, 41, 2):
    p = np.percentile(D, i)
    _, _, obj_val, _ = classifier.cross_val(X, y, p, 1, 4, False)
    obj_vals.append(obj_val)

ps = list(list(range(2, 41, 2)))

plt.plot(ps, obj_vals)
plt.show()
