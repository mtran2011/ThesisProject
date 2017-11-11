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
            ndarray: self._params
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
        ''' Estimate q(s,a)
        Args:
            state (iterable): ndarray, list or tuple of state features
            action (float): a scalar value for the action
        Returns:
            float: the value of q(s,a)
        '''
        pass
    
    @abc.abstractmethod
    def eval_gradient(self, state, action):
        ''' Evaluate the gradient with respect to the params
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
        the linear coefs:
            n+1 params for action a and K state features S(1) to S(n)
            n params for product of a and each S(i)
            and so on
        the constant coefs:
            same shape appended along axis=1
        '''
        # the left part of self._params is an upper triangular matrix for linear coefs
        self._params = np.ones((num_state_features+1, num_state_features+1)) / 100.0
        self._params = np.triu(self._params, k=0)
        # the left part of self._params is an upper triangular matrix for constant coefs
        const = self._params * 1
        self._params = np.concatenate((self._params, const), axis=1)
    
    def _make_input_matrix(self, state, action):
        ''' Turn the inputs (s,a) into an upper triangular matrix of pairwise product
        Args:
            state (iterable): ndarray, list or tuple of state features
            action (float): a scalar value for the action
        Returns:
            ndarray: shape (n+1,n+1) where n is len(state)
        '''
        n = len(state)
        if n != (self._params.shape[0]-1):
            raise ValueError('the length of state input is inconsistent')

        inputs = [action, *state]
        # set up the input matrix of pairwise product, this is upper triangular
        input_matrix = np.zeros((n+1, n+1))        
        for i in range(n+1):
            for j in range(i,n+1):
                if i != j:
                    input_matrix[i,j] = inputs[i] * inputs[j]
                else:
                    input_matrix[i,j] = inputs[i]
        return input_matrix

    # Override base class abstractmethod
    def estimate_q(self, state, action):
        input_matrix = self._make_input_matrix(state, action)
        n = len(state)        
        q = 0
        for i in range(n+1):            
            q += np.dot(input_matrix[i,:], self._params[i,:n+1]) + np.sum(self._params[i,n+1:])
        return q
    
    # Override base class abstractmethod
    def eval_gradient(self, state, action):
        input_matrix = self._make_input_matrix(state, action)
        n = len(state)
        constant = np.triu(np.ones((n+1, n+1)))
        grad = np.concatenate((input_matrix, constant), axis=1)
        assert not np.isnan(grad).any()
        return grad