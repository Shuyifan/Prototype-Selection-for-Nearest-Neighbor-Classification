from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from classifier import classifier

iris = load_iris()
X = iris.data
y = iris.target

classifier.cross_val(X, y, 0.5, 1, 4, False)