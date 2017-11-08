import abc
import numpy as np 

class QFunctionEstimator(metaclass=abc.ABCMeta):
    ''' Abstract base class for a parametric function estimator of q(s,a)
    Attributes:
        _params (ndarray): the parameters used in estimation
    '''    
        
    def get_params(self):
        '''
        Returns:
            ndarray: return self._params
        '''
        return self._params
        
    def set_params(self, params):
        '''
        Args:
            params (ndarray): params to be set
        Returns:
            None: set self._params internally
        '''
        if params.shape != self._params.shape:
            raise ValueError('params ndarray shape is wrong')
        self._params = params
    
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
                     must have the same shape as self._params
        '''
        pass

class CubicEstimator(QFunctionEstimator):
    ''' Parametric cubic function estimator of q(s,a)
    Attributes:
        _params (ndarray): the cubic coefficients serve as parameters
    '''
    def __init__(self, num_state_features):
        ''' Initialize self._params to a 4 x (num_state_features + 3) matrix
        The first num_state_features columns: 4 cubic coefs for each state feature
        Next column: 4 cubic coefs for the product of all state features
        Next column: 4 cubic coefs for scalar-valued action a
        Next column: 4 cubic coefs for the product of all state features x a
        Args:
            num_state_features (int): number of features in a state
        '''
        self._params = np.ones((4, num_state_features + 3)) / 1e3
    
    # Override base class abstractmethod
    def estimate_q(self, state, action):
        n, q = len(state), 0
        if n != (self._params.shape[1]-3):
            raise ValueError('the length of state input is wrong')
        for i in range(n):
            q += np.polyval(self._params[:,i], state[i])
        x = np.prod(state)
        q += np.polyval(self._params[:,n], x)
        q += np.polyval(self._params[:,n+1], action)
        q += np.polyval(self._params[:,n+2], action*x)
        return q
    
    # Override base class abstractmethod
    def eval_gradient(self, state, action):
        n = len(state)
        if n != (self._params.shape[1]-3):
            raise ValueError('the length of state input is wrong')
        grad = np.zeros(self._params.shape)
        for row in range(4):
            for col in range(n):
                grad[row,col] = state[col]**(3-row)
        x = np.prod(state)
        for row in range(4):
            grad[row,n] = x**(3-row)
            grad[row,n+1] = action**(3-row)
            grad[row,n+2] = (action*x)**(3-row)
        return grad

class PairwiseLinearEstimator(QFunctionEstimator):
    def __init__(self, num_state_features):
        ''' n = num_state_features, and the params consist of:
        n+1 params for action a and K state features S(1) to S(n)
        n params for product of a and each S(i)
        and so on
        '''
        # self._params is an upper triangular matrix
        self._params = np.ones((num_state_features+1, num_state_features+1)) / 1e2
        self._params = np.triu(self._params)
        # todo: doesn't have the const coef in each linear func yet
    
    # Override base class abstractmethod
    def estimate_q(self, state, action):
        n = len(state)
        if n != (self._params.shape[0]-1):
            raise ValueError('the length of state input is wrong')
        inputs = [action, *state]
        input_matrix = np.zeros((num_state_features+1, num_state_features+1))
        pass

        