'''Libraries for Prototype selection'''
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
import cvxpy as cvx
import math as mt
from sklearn.model_selection import KFold
import sklearn.metrics


class classifier():
    '''Contains functions for prototype selection'''
    def __init__(self, X, y, epsilon_, lambda_ ):
        '''Store data points as unique indexes, and initialize 
        the required member variables eg. epsilon, lambda, 
        interpoint distances, points in neighborhood'''
        self.X = X
        self.y = y
        self.epsilon = epsilon_
        self.lambda_ = lambda_

    '''Implement modules which will be useful in the train_lp() function
    for example
    1) operations such as intersection, union etc of sets of datapoints
    2) check for feasibility for the randomized algorithm approach
    3) compute the objective value with current prototype set
    4) fill in the train_lp() module which will make 
    use of the above modules and member variables defined by you
    5) any other module that you deem fit and useful for the purpose.'''

    
    def train_lp(self, verbose = False):
        '''Implement the linear programming formulation 
        and solve using cvxpy for prototype selection'''
        
    def objective_value(self):
        '''Implement a function to compute the objective value of the integer optimization
        problem after the training phase'''
        
    def predict(self, instances):
        '''Predicts the label for an array of instances using the framework learnt'''


def cross_val(data, target, epsilon_, lambda_, k, verbose):
    '''Implement a function which will perform k fold cross validation 
    for the given epsilon and lambda and returns the average test error and number of prototypes'''
    kf = KFold(n_splits=k, random_state = 42)
    score = 0
    prots = 0
    for train_index, test_index in kf.split(data):
        ps = classifier(data[train_index], target[train_index], epsilon_, lambda_)
        ps.train_lp(verbose)
        obj_val += ps.objective_value()
        score += sklearn.metrics.accuracy_score(target[test_index], ps.predict(data[test_index]))
        '''implement code to count the total number of prototypes learnt and store it in prots'''
    score /= k    
    prots /= k
    obj_val /= k
    return score, prots, obj_val
