import numpy as np
import cvxpy as cvx
import math as mt
from sklearn.model_selection import KFold
import sklearn.metrics

class classifier():
    """
    This class implements a nearest neighbor classification with prototype selection.
    The class contains the function with initialization, training, and predction.

    :attribute X: the training dataset
    :attribute y: the ground truth label of corresponding data in X
    :attribute epsilon: the hyperparameter for the prototype selection, which indicates 
                        the radius of region which is coverd by a prototype.
    :attribute lambda: the hyperparameter for the prototype selection during the
                       optimization part.
    :attribute alpha_l: the optimization result (alpha) of the LP prblem for each label (l).
    :attribute xi_l: the optimization result (xi) of the LP prblem for each label (l).
    :attribute Alpha_l: the optimization result (Alpha) for each label (l) after using randomized 
                        rounding approach to get a solution for the ILP.
    :attribute Xi_l: the optimization result (Xi) for each label (l) after using randomized 
                     rounding approach to get a solution for the ILP.
    :attribute prots: the number of prototypes selected.

    :type X: numpy.ndarray (shape: (n, p))
    :type y: numpy.ndarray (shape: (n, ))
    :type epsilon: float
    :type lambda: float
    :type alpha_l: list (length: l and each element inside it is numpy.ndarray.)
    :type xi_l: list (length: l and each element inside it is numpy.ndarray.)
    :type Alpha_l: list (length: l and each element inside it is numpy.ndarray.)
    :type Xi_l: list (length: l and each element inside it is numpy.ndarray.)
    :type prots: int
    """


    def __init__(self, X, y, epsilon_, lambda_):
        """
        The init function for the class. Initialize the needed attribute.
        
        :param X: the training dataset
        :param y: the ground truth label of corresponding data in X
        :param epsilon_: the hyperparameter for the prototype selection, which indicates 
                         the radius of region which is coverd by a prototype.
        :param lambda_: the hyperparameter for the prototype selection during the
                        optimization part.

        :type X: numpy.ndarray (shape: (n, p))
        :type y: numpy.ndarray (shape: (n, ))
        :type epsilon_: float
        :type lambda_: float
        """
        self.X = X
        self.y = y
        self.epsilon = epsilon_
        self.lambda_ = 1.0 / X.shape[0]
        self.alpha_l = None
        self.xi_l = None
        self.Alpha_l = None
        self.Xi_l = None
        self.prots = None
        
        self.__LP_object_l = None
        self.__Cl_j = None
        self.__M_l = None
        self.__y_labels = None
        self.__Xl_index_set = None
        self.__X1_index_list = None
        self.__Xl = None
        self.__region = None

        self.__init_Xl()
        self.__cal_region_set()

    def train_lp(self, verbose = False):
        """
        The function to train the model using the class attribute.
        
        :param verbose: a boolean value used for the debugging process. If it is True,
                        the function will show all the information during the training 
                        step; if not, it will show nothing.

        :type verbose: bool
        """
        self.alpha_l = list()
        self.xi_l = list()
        self.__Cl_j = list()
        self.__M_l = list()
        self.__LP_object_l = list()

        for i in range(len(self.__Xl)):
            alpha = cvx.Variable((self.X.shape[0], 1))
            xi = cvx.Variable((self.__Xl[i].shape[0], 1))
            
            C_j = self.__calc_Cl_j(i)
            M = self.__calc_M(i)

            self.__Cl_j.append(C_j)
            self.__M_l.append(M)

            constraints = [alpha >= 0, alpha <= 1, xi >= 0, M * alpha >= 1 - xi]
        
            obj = cvx.Minimize(sum(alpha.T * C_j) + sum(xi))

            prob = cvx.Problem(obj, constraints)
            prob.solve(verbose = verbose)

            self.alpha_l.append(alpha)
            self.xi_l.append(xi)
            self.__LP_object_l.append(prob.value)
        
        self.Alpha_l = list()
        self.Xi_l = list()
        self.prots = 0
        for i in range(len(self.alpha_l)):
            Alpha = np.zeros((self.X.shape[0], 1))
            Xi = np.zeros((self.__Xl[i].shape[0], 1))
            while(not self.__is_feasible(Alpha, Xi, i)):
                for _t in range(2 * mt.ceil(np.log2(self.__Xl[i].shape[0]))):
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
                    if(self.__is_feasible(Alpha, Xi, i)):
                        break
            self.prots = self.prots + np.sum(Alpha)
            self.Alpha_l.append(Alpha)
            self.Xi_l.append(Xi)

    def objective_value(self):
        """
        A function to compute the objective value of the integer optimization problem after 
        the training phase.

        :returns optimal_value: the objective value of the integer optimization problem after 
                                the training phase.

        :rtype optimal_value: float
        """
        optimal_value = 0
        for i in range(len(self.Alpha_l)):
            optimal_value = optimal_value + np.sum(np.dot(self.Alpha_l[i].T, self.__Cl_j[i])) + np.sum(self.Xi_l[i])
        return optimal_value


    def predict(self, instances):
        """
        Given an unseen data, this function is used to predict the label using the model we trained.
        
        :param instances: the unseen input dataset.

        :type instances: numpy.ndarray (shape: (n, ))

        :returns pred_y: the prediction labels.

        :rtype pred_y: numpy.ndarray (shape: (n, ))
        """
        pred_y = np.zeros((instances.shape[0], ))
        for i in range(pred_y.shape[0]):
            min_distance_l = float("inf")
            pred_label_index = 0
            for label in range(len(self.__y_labels)):
                min_distance = float("inf")
                for j in range(self.Alpha_l[label].shape[0]):
                    if(self.Alpha_l[label][j, 0] == 1):
                        min_distance = min(min_distance, np.linalg.norm(instances[i, : ] - self.X[j, : ]))
                if(min_distance < min_distance_l):
                    min_distance_l = min_distance
                    pred_label_index = label
            pred_y[i] = self.__y_labels[pred_label_index]
        return pred_y
    
    @staticmethod
    def cross_val(data, target, epsilon_, lambda_, k, verbose):
        """
        A static method to use a K-folder method to evaluate the performance of 
        the nearest neighbor classifier with prototype selection using different 
        hyperparameters.
        
        :param data: the input dataset.
        :param target: the ground truth label of the input dataset.
        :param epsilon_: the hyperparameter for the prototype selection, which indicates 
                         the radius of region which is coverd by a prototype.
        :param lambda_: the hyperparameter for the prototype selection during the
                        optimization part.
        :param k: the number of folder we split for the K-folder validation method.
        :param verbose: a boolean value used for the debugging process. If it is True,
                        the function will show all the information during the training 
                        step; if not, it will show nothing.

        :type data: numpy.ndarray (shape: (n, p))
        :type target: numpy.ndarray (shape: (n, ))
        :type epsilon_: float
        :type lambda_: float
        :type k: int
        :type verbose: bool

        :returns score: the average accuracy of the model after using K-folder validation.
        :returns prots: the average number of prototypes the model generated 
                        using K-folder validation.
        :returns obj_val: the average objective value of the model after using K-folder validation.

        :rtype score: float
        :rtype prots: float
        :rtype obj_val: float
        """
        kf = KFold(n_splits=k, random_state = 42)
        score = 0
        prots = 0
        obj_val = 0
        for train_index, test_index in kf.split(data):
            ps = classifier(data[train_index], target[train_index], epsilon_, lambda_)
            ps.train_lp(verbose)
            obj_val += ps.objective_value()
            score += sklearn.metrics.accuracy_score(target[test_index], ps.predict(data[test_index]))
            prots += ps.prots
        score /= k    
        prots /= k
        obj_val /= k
        return score, prots, obj_val

    def __init_Xl(self):
        self.__y_labels = set()
        for i in range(self.y.shape[0]):
            self.__y_labels.add(self.y[i])
        self.__y_labels = list(self.__y_labels)

        self.__Xl_index_set = list()
        self.__X1_index_list = list()
        
        for i in range(len(self.__y_labels)):
            index = set()
            for j in range(self.y.shape[0]):
                if(self.y[j] == i):
                    index.add(j)
            self.__X1_index_list.append(list(index))
            self.__Xl_index_set.append(index)

        self.__Xl = list()
        for i in range(len(self.__y_labels)):
            self.__Xl.append(self.X[self.__X1_index_list[i], : ])

    def __cal_region_set(self):
        self.__region = list()
        for i in range(self.X.shape[0]):
            B_xj = set()
            for j in range(self.X.shape[0]):
                if(np.linalg.norm(self.X[i , : ] - self.X[j, : ]) < self.epsilon):
                    B_xj.add(j)
            self.__region.append(B_xj)

    def __calc_Cl_j(self, l):
        Cl_j = np.zeros((self.X.shape[0], 1))
        for i in range(Cl_j.shape[0]):
            Cl_j[i, 0] = self.lambda_ + len(self.__region[i] & (set(range(self.X.shape[0])) - self.__Xl_index_set[l]))
        return Cl_j

    def __calc_M(self, l):
        M = np.zeros((self.__Xl[l].shape[0], self.X.shape[0]))
        for i in range(self.__Xl[l].shape[0]):
            for j in range(self.X.shape[0]):
                if(self.__X1_index_list[l][i] in self.__region[j]):
                    M[i, j] = 1
        return M

    def __is_feasible(self, Alpha, Xi, l):
        if(np.any(np.dot(self.__M_l[l], Alpha) < 1 - Xi)):
            return False
        if(np.sum(np.dot(Alpha.T, self.__Cl_j[l])) + np.sum(Xi) > 2 * np.log2(Xi.shape[0]) * self.__LP_object_l[l]):
            return False
        return True