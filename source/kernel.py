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
        Returns:
            float: must be scalar, the predicted value
        '''
        raise NotImplementedError

class InverseNormWeighter(KernelSmoothingRegressor):
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
        for i in range(X.shape[0]):
            x0 = X[i]
            # estimate f(x0)
            d_vals = np.array([1 / np.linalg.norm(x_i - x0, ord=self.p) for x_i in self.X])
            assert d_vals.shape[0] == self.Y.shape[0]
            f_x0 = np.asscalar(np.dot(d_vals, self.Y) / d_vals.sum())
            f_vals.append(f_x0)
        return np.array(f_vals)