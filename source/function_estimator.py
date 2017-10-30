import abc
import numpy as np 

class QFunctionEstimator(object):
    ''' Abstract base class for a parametric function estimator of q(s,a)
    Attributes:
        _params (ndarray): the parameters used in estimation
    '''
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def get_params(self):
        '''
        Returns:
            ndarray: return self._params
        '''
        pass
    
    @abc.abstractmethod
    def set_params(self, params):
        '''
        Args:
            params (ndarray): params to be set
        Returns:
            None: set self._params internally
        '''
        pass
    
    @abc.abstractmethod
    def estimate_q(self, state, action):
        '''
        Args:
            state (iterable): ndarray, list or tuple of state features
            action (float): a scalar value for the action
        Returns:
            float: the value of q(s,a)
        '''
        pass
    
    @abc.abstractmethod
    def eval_gradient(self, state, action):
        '''
        Args:
            state (iterable): ndarray, list or tuple of state features
            action (float): a scalar value for the action
        Returns:
            ndarray: the gradient with respect to self._params, evaluated at (s,a)
        '''
        pass

class CubicEstimator(QFunctionEstimator):
    ''' Parametric cubic function estimator of q(s,a)
    Attributes:
        _params (ndarray): the cubic coefficients serve as parameters
    '''
    def __init__(self, num_state_features):
        '''
        Args:
            num_state_features (int): number of features in a state
        '''
        # 4 coefs for each state feature
        # 4 coefs for the cubic applied on product of state features
        # 4 coefs for the cubic on the scalar action
        # 4 coefs for the cubic on the product of a and all state features
        self._num_state_features = num_state_features
        self._params = np.zeros(4 * num_state_features + 12)

    # Override base class abstractmethod
    def estimate_q(self, state, action):        
        m, q = len(state), 0
        if m != self._num_state_features:
            raise ValueError('the length if state input is wrong')
        for i in range(m):
            q += np.polyval(self._params[4*i:4*i+4], state[i])
        x = np.prod(state)
        q += np.polyval(self._params[4*m:4*m+4], x)
        q += np.polyval(self._params[4*m+4:4*m+8], action)
        q += np.polyval(self._params[4*m+8:4*m+12], action*x)
        return q