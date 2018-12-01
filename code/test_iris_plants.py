from sklearn.datasets import load_iris
from classifier import classifier

iris = load_iris()
X = iris.data
y = iris.target

nnClassifier = classifier(X, y, 0.5, 1)

nnClassifier.train_lp()