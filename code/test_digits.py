from sklearn.datasets import load_digits
from scipy.spatial import distance_matrix
import numpy as np
from classifier import classifier
import matplotlib.pyplot as plt

digits = load_digits()

X = digits.data
y = digits.target

D = distance_matrix(X, X)

p_list = list()
obj_list = list()
for i in range(2, 41, 2):
    p = np.percentile(D, i)
    p_list.append(p)
    _, _, obj, _ = classifier.cross_val(X, y, p, 1, 4, False)
    obj_list.append(obj)
    print(i)

plt.figure(1)
new_p_list, new_obj_list = zip(*sorted(zip(p_list, obj_list)))
plt.plot(new_p_list, new_obj_list)
plt.show(block=True)