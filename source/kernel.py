''' Module for kernel smoothing regressor
'''
import abc
import numpy as np

class KernelSmoothingRegressor(abc.ABC):
    ''' Abstract base class for a kernel smoothing regressor
    Attributes:
        X (ndarray): array of N x p for N observations, p features
        Y (ndarray): array of N x 1 for N observations, each y is a float
    '''
    def __init__(self):
        self.X = None
        self.Y = None
    
    def fit(self,X,Y):
        ''' Fit training data
        '''
        if not isinstance(X, numpy.ndarray) or not isinstance(Y, numpy.ndarray):
            raise ValueError('X and Y must be ndarray')
        if X.shape[0] != Y.shape[0]:
            raise ValueError('X and Y must have same first dimension')
        self.X = X
        self.Y = Y
    
    @abc.abstractmethod
    def predict(self, X):
        ''' Predict Y for given X
        '''
        raise NotImplementedError

class NormBasedSmoother(KernelSmoothingRegressor):
    '''
    '''

def inverse_norm_p(x1, x2, p=2):
    '''
    Args:
        x1 (list): list of coordinates of the first point
        x2 (list): list of coordinates of the second point
    Returns:
        float: inverse of distance in L-p norm between x1 and x2
    '''
    x1 = np.array(x1)
    x2 = np.array(x2)
    return 1 / np.linalg.norm(x1 - x2, ord=p)