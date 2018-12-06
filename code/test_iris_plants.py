from sklearn.datasets import load_iris
from classifier import classifier
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

score_2, prots_2, obj_val_2, cover_error_2 = classifier.cross_val(X, y, 0.5, 1, 4, False, norm = 2)
score_1, prots_1, obj_val_1, cover_error_1 = classifier.cross_val(X, y, 0.5, 1, 4, False, norm = 1)
score_inf, prots_inf, obj_val_inf, cover_error_inf = classifier.cross_val(X, y, 0.5, 1, 4, False, norm = np.inf)