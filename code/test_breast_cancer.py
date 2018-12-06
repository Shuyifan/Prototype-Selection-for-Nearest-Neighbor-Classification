from sklearn.datasets import load_breast_cancer
from scipy.spatial import distance_matrix
import numpy as np
from classifier import classifier
import matplotlib.pyplot as plt
import numpy as np

breast_cancer = load_breast_cancer()

X = breast_cancer.data
y = breast_cancer.target

D = distance_matrix(X, X, p = np.inf)

p_list = list()
test_errors = list()
cover_errors = list()
prots_list = list()
for i in range(2, 41, 2):
    p = np.percentile(D, i)
    p_list.append(p)
    score, prots, _, cover_error = classifier.cross_val(X, y, p, 1, 4, False, norm = np.inf)
    test_error = 1.0 - score
    test_errors.append(test_error)
    cover_errors.append(cover_error)
    prots_list.append(prots)
    print(i)

ps = list(list(range(2, 41, 2)))

plt.figure(1)
plt.plot(p_list, test_errors)
plt.show()

plt.figure(2)
plt.plot(p_list, cover_errors)
plt.show()

plt.figure(3)
new_ps, new_test_errors = zip(*sorted(zip(ps, test_errors)))
plt.plot(new_ps, new_test_errors)
plt.show()

plt.figure(4)
new_ps, new_cover_errors = zip(*sorted(zip(ps, cover_errors)))
plt.plot(new_ps, new_cover_errors)
plt.show()

plt.figure(5)
new_prots_list, new_test_errors = zip(*sorted(zip(prots_list, test_errors)))
plt.plot(new_prots_list, new_test_errors)
plt.show()

plt.figure(6)
new_prots_list, new_cover_errors = zip(*sorted(zip(prots_list, cover_errors)))
plt.plot(new_prots_list, new_cover_errors)
plt.show(block=True)