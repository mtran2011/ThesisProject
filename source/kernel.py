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
        assert isinstance(X, np.ndarray) and isinstance(Y, np.ndarray)
        assert X.shape[0] == Y.shape[0]            
        assert Y.shape[0] == Y.size
        self.X = X
        self.Y = Y
    
    @abc.abstractmethod
    def predict(self, X):
        ''' Predict Y for given X
        '''
        raise NotImplementedError

class InverseNormAverage(KernelSmoothingRegressor):
    ''' Weighted average with inverse distance using norm p = 1 or p = 2
    '''
    def __init__(self, p):
        if p not in [1,2]:
            raise ValueError('norm-based kernel should use p=1 or p=2 only')
        super().__init__()
        self.p = p
    
    # Override
    def predict(self, X):
        if self.X is None or self.Y is None:
            raise ValueError('must have training data first')
        f_vals = []
        for x0 in X:
            # estimate f(x0)
            d_vals = np.array([1 / np.linalg.norm(x_i - x0, ord=self.p) for x_i in self.X])
            f_x0 = np.asscalar(np.dot(d_vals, self.Y) / d_vals.sum())
            f_vals.append(f_x0)
        return np.array(f_vals)

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