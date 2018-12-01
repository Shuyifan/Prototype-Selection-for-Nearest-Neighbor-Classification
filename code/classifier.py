'''Libraries for Prototype selection'''
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
import cvxpy as cvx
import math as mt
from sklearn.model_selection import KFold
import sklearn.metrics

class classifier():
    '''Contains functions for prototype selection'''
    def __init__(self, X, y, epsilon_, lambda_):
        '''Store data points as unique indexes, and initialize 
        the required member variables eg. epsilon, lambda, 
        interpoint distances, points in neighborhood'''
        self.X = X
        self.y = y
        self.epsilon = epsilon_
        self.lambda_ = 1.0 / X.shape[0]
        
        self.init_Xl()
        self.cal_region_set()

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
        self.alpha_l = list()
        self.xi_l = list()
        self.Cl_j = list()
        self.M_l = list()
        self.LP_object_l = list()

        for i in range(len(self.Xl)):
            alpha = cvx.Variable((self.X.shape[0], 1))
            xi = cvx.Variable((self.Xl[i].shape[0], 1))
            
            C_j = self.calc_Cl_j(i)
            M = self.calc_M(i)

            self.Cl_j.append(C_j)
            self.M_l.append(M)

            constraints = [alpha >= 0, alpha <= 1, xi >= 0, M * alpha >= 1 - xi]
        
            obj = cvx.Minimize(sum(alpha.T * C_j) + sum(xi))

            prob = cvx.Problem(obj, constraints)
            prob.solve(verbose = verbose)

            self.alpha_l.append(alpha)
            self.xi_l.append(xi)
            self.LP_object_l.append(prob.value)
        
        self.Alpha_l = list()
        self.Xi_l = list()
        for i in range(len(self.alpha_l)):
            Alpha = np.zeros((self.X.shape[0], 1))
            Xi = np.zeros((self.Xl[i].shape[0], 1))
            while(not self.is_feasible(Alpha, Xi, i)):
                for _t in range(2 * mt.ceil(np.log2(self.Xl[i].shape[0]))):
                    temp_alpha = self.alpha_l[i].value
                    temp_xi = self.xi_l[i].value
                    temp_alpha[temp_alpha < 0] = 0
                    temp_alpha[temp_alpha > 1] = 1
                    temp_xi[temp_xi < 0] = 0
                    temp_xi[temp_xi > 1] = 1
                    A = np.random.binomial(1, p = temp_alpha)
                    X = np.random.binomial(1, p = temp_xi)
                    Alpha = np.maximum(Alpha, A)
                    Xi = np.maximum(Xi, X)
                    if(self.is_feasible(Alpha, Xi, i)):
                        break
            self.Alpha_l.append(Alpha)
            self.Xi_l.append(Xi)

    def objective_value(self):
        '''Implement a function to compute the objective value of the integer optimization
        problem after the training phase'''
        


    def predict(self, instances):
        '''Predicts the label for an array of instances using the framework learnt'''

    def init_Xl(self):
        y_labels = set()
        for i in range(self.y.shape[0]):
            y_labels.add(self.y[i])
        y_labels = list(y_labels)

        self.Xl_index_set = list()
        self.X1_index_list = list()
        for i in range(len(y_labels)):
            index = set()
            for j in range(self.y.shape[0]):
                if(self.y[j] == i):
                    index.add(j)
            self.X1_index_list.append(list(index))
            self.Xl_index_set.append(index)

        self.Xl = list()
        for i in range(len(y_labels)):
            self.Xl.append(self.X[self.X1_index_list[i], : ])

    def cal_region_set(self):
        self.region = list()
        for i in range(self.X.shape[0]):
            B_xj = set()
            for j in range(self.X.shape[0]):
                if(np.linalg.norm(self.X[i , : ] - self.X[j, : ]) < self.epsilon):
                    B_xj.add(j)
            self.region.append(B_xj)

    def calc_Cl_j(self, l):
        Cl_j = np.zeros((self.X.shape[0], 1))
        for i in range(Cl_j.shape[0]):
            Cl_j[i, 0] = self.lambda_ + len(self.region[i] & (set(range(self.X.shape[0])) - self.Xl_index_set[l]))
        return Cl_j

    def calc_M(self, l):
        M = np.zeros((self.Xl[l].shape[0], self.X.shape[0]))
        for i in range(self.Xl[l].shape[0]):
            for j in range(self.X.shape[0]):
                if(self.X1_index_list[l][i] in self.region[j]):
                    M[i, j] = 1
        return M

    def is_feasible(self, Alpha, Xi, l):
        if(np.any(np.dot(self.M_l[l], Alpha) < 1 - Xi)):
            return False
        if(np.sum(np.dot(Alpha.T, self.Cl_j[l])) + np.sum(Xi) > 2 * np.log2(Xi.shape[0]) * self.LP_object_l[l]):
            return False
        return True

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
