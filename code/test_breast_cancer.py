from sklearn.datasets import load_breast_cancer
from scipy.spatial import distance_matrix
import numpy as np
from classifier import classifier
import matplotlib.pyplot as plt

breast_cancer = load_breast_cancer()

X = breast_cancer.data
y = breast_cancer.target

D = distance_matrix(X, X)

test_errors = list()
cover_errors = list()
for i in range(2, 41, 2):
    p = np.percentile(D, i)
    score, _, _, cover_error = classifier.cross_val(X, y, p, 1, 4, False)
    test_error = 1.0 - score
    test_errors.append(test_error)
    cover_errors.append(cover_error)

ps = list(list(range(2, 41, 2)))

plt.plot(ps, test_error)
plt.show()

plt.plot(ps, cover_error)
plt.show()