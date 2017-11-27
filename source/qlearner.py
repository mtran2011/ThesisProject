import abc
import random
import numpy as np
from learner import Learner

# Adapted from:
# github.com/vmayoral/basic_reinforcement_learning/blob/master/tutorial1/qlearn.py
# keon.io/deep-q-learning/

class QLearner(Learner):
    ''' Abstract base class for a Q-learning agent    
    '''
    
    # Override base class abstractmethod
    def learn(self, reward, new_state):
        ''' Get a reward and see a new_state. Use these to do some internal training. Then return a new action.        
        Update _last_state <- new_state
        Update _last_action <- best_action        
        Args:
            reward (float): the reward seen after the previous action
            new_state (tuple): the new_state seen after the previous action
        Returns:
            best_action (object): take a new action based on new_state
        '''
        # if this agent has taken at least one any action before
        if self._last_action is not None and self._last_state is not None:
            self._train_internally(reward, new_state)
        action = self._find_action_greedily(new_state)
        self._last_action = action
        self._last_state = new_state
        return action

class QMatrix(QLearner):
    ''' Abstract class for a Q-learner matrix that holds the values of Q(s,a)
    Attributes:
        _Q (dict): dict of key tuple (s,a) to float value Q(s,a)        
        _epsilon (float): constant in epsilon-greedy policy
        _learning_rate (float): the constant learning_rate
        _discount_factor (float): the constant discount_factor of future rewards        
    '''
    
    def __init__(self, actions, epsilon, learning_rate, discount_factor):
        super().__init__(actions)
        self._Q = dict()
        self._epsilon = epsilon
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
    
    @abc.abstractmethod
    def _get_q(self, state, action):
        ''' Find, or estimate, Q(s,a) for a given state s and action a
        Args:
            state (tuple): state s, a tuple of state attributes
            action (object): action a
        Returns:
            float: the value of Q(s,a). 
        '''
        pass        
        
    # Override base class abstractmethod
    def _find_action_greedily(self, state, use_epsilon=True, return_q=False):        
        if use_epsilon and random.random() < self._epsilon:
            # with probability epsilon, choose from all actions with equal chance
            best_action = random.choice(self._actions)
            max_q = None
            if return_q:
                max_q = self._get_q(state, best_action)
        else:
            # choose action = arg max {action} of Q(state, action)
            q_values = [self._get_q(state, action) for action in self._actions]
            max_q = max(q_values)
            best_action = self._actions[q_values.index(max_q)]
        
        if return_q:
            return best_action, max_q
        else:
            return best_action       
    
    # Override base class abstractmethod
    def _train_internally(self, reward, new_state):
        # cannot train if never seen a state or took an action before
        if self._last_action is None or self._last_state is None:
            return None
        
        # use reward, new_state to update Q(self._last_state, self._last_action)
        old_q = self._get_q(self._last_state, self._last_action)
        # find max over a of Q(s',a)
        _, max_q = self._find_action_greedily(new_state, use_epsilon=False, return_q=True)
        new_q = old_q + self._learning_rate * (reward + self._discount_factor * max_q - old_q)
        self._Q[(self._last_state, self._last_action)] = new_q        

class TabularQMatrix(QMatrix):
    # Override base class abstractmethod
    def _get_q(self, state, action):
        return self._Q.get((state, action), 0)

class KernelSmoothingQMatrix(QMatrix):
    ''' Use kernel smoothing to estimate Q(s,a) if this has not been found before
    Attributes:
        _kernel_func (function): to calculate K(x1,x2) = D(||x1-x2||) where x1 = (s1, a1) and x2 = (s2, a2)
        _sample_size (int): how many samples to take from existing Q(s,a) as training data for kernel smoother
    '''

    def __init__(self, actions, kernel_func, epsilon, learning_rate, discount_factor, sample_size=30):
        super().__init__(actions, epsilon, learning_rate, discount_factor)
        self._kernel_func = kernel_func
        self._sample_size = sample_size
    
    # Override base class abstractmethod
    def _get_q(self, state, action):        
        if not self._Q:
            return 0
        if (state, action) in self._Q:
            return self._Q[(state, action)]
        
        # now that (state, action) is not in Q, try to estimate Q(state, action) via smoothing
        # first sample a training data from existing Q(s,a)
        sample_size = min(self._sample_size, len(self._Q))
        batch = random.sample(self._Q.keys(), sample_size)

        k_vals, q_vals = [], []
        x0 = [*state, action] 
        for that_state, that_action in batch:
            x2 = [*that_state, that_action]
            k_vals.append(self._kernel_func(x0, x2))
            q_vals.append(self._Q[(that_state, that_action)])
        
        k_vals = np.array(k_vals)
        q_vals = np.array(q_vals)
        estimate = np.asscalar(k_vals.dot(q_vals) / k_vals.sum())
        self._Q[(state, action)] = estimate
        return estimate

class DQNLearner(QLearner):
    ''' Class for a Q-learner that uses a network to estimate Q(s,a) for all a
    Attributes:        
        _epsilon (float): constant in epsilon-greedy policy        
        _discount_factor (float): the constant discount_factor of future rewards
        _memory (list): list of tuples (s,a,r,s') to store for experience replay
        _model (Sequential): a network using keras Sequential
    '''
    
    def __init__(self, actions, model, epsilon, discount_factor):
        super().__init__(actions)
        self._epsilon = epsilon        
        self._discount_factor = discount_factor
        self._memory = []
        self._model = model # todo, should model be an input?
        
    # Override base class abstractmethod
    def _find_action_greedily(self, state, use_epsilon=True, return_q=False):
        # translate state from tuple to ndarray
        state = np.array(state).reshape(1, len(state))
        
        if use_epsilon and random.random() < self._epsilon:
            # with probability epsilon, choose from all actions with equal chance
            best_action = random.choice(self._actions)
            max_q = None
            if return_q:
                # get all the Q(s,a) for this state
                # model.predict(X) where X is a n x p array of n obs, each ob having p features
                # model.predict(X) gives an array of predictions; each prediction is an array of size k
                q_values = self._model.predict(state)[0]
                max_q = q_values[self._actions.index(best_action)]
        else:
            # get all the Q(s,a) for this state            
            q_values = self._model.predict(state)[0]
            index = np.argmax(q_values)
            max_q = q_values[index]
            best_action = self._actions[index]
        
        if return_q:
            return best_action, max_q
        else:
            return best_action
    
    def _replay(self, sample_size=30):
        ''' Sample from memory, a collection of (s,a,r,s'), to train internal model
        Args:
            sample_size (int): how many samples to draw from memory
        Returns:
            None: only train the model internally
        '''
        sample_size = min(sample_size, len(self._memory))
        batch = random.sample(self._memory, sample_size)
        
        for state, action, reward, next_state in batch:
            # translate state from tuple to ndarray
            state = np.array(state).reshape(1, len(state))
            next_state = np.array(next_state).reshape(1, len(next_state))
            # target value for this specific action
            target_for_a = reward + self._discount_factor * np.amax(self._model.predict(next_state)[0])
            # target value for all actions except this specific action
            target_for_all_a = self._model.predict(state)
            target_for_all_a[0][self._actions.index(action)] = target_for_a
            self._model.fit(state, target_for_all_a, verbose=0)
        return None
    
    # Override base class abstractmethod
    def _train_internally(self, reward, new_state):
        if self._last_action is None or self._last_state is None:
            return None
        # memorize this experience and then train the neural network
        self._memory.append((self._last_state, self._last_action, reward, new_state))
        self._replay()        

class SemiGradQLearner(QLearner):
    ''' Class for a Q-learner that uses a parametric function approximator to estimate Q(s,a) for all a
    Attributes:
        _epsilon (float): constant in epsilon-greedy policy
        _learning_rate (float): the constant learning_rate
        _discount_factor (float): the constant discount_factor of future rewards        
        _estimator (QFunctionEstimator): a parametric function estimator
    '''
    
    def __init__(self, actions, estimator, epsilon, learning_rate, discount_factor):
        super().__init__(actions)
        self._epsilon = epsilon
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._estimator = estimator # the function estimator
    
    # Override base class abstractmethod
    def _find_action_greedily(self, state, use_epsilon=True, return_q=False):        
        if use_epsilon and random.random() < self._epsilon:
            # with probability epsilon, choose from all actions with equal chance
            best_action = random.choice(self._actions)
            max_q = None
            if return_q:
                max_q = self._estimator.estimate_q(state, best_action)
        else:
            q_values = [self._estimator.estimate_q(state, action) for action in self._actions]
            max_q = max(q_values)
            best_action = self._actions[q_values.index(max_q)]
        
        if return_q:
            return best_action, max_q
        else:
            return best_action
    
    # Override base class abstractmethod
    def _train_internally(self, reward, new_state):
        if self._last_action is None or self._last_state is None:
            return None

        old_q = self._estimator.estimate_q(self._last_state, self._last_action)
        _, max_q = self._find_action_greedily(new_state, use_epsilon=False, return_q=True)        
        # gradient with respect to the parameters
        grad = self._estimator.eval_gradient(self._last_state, self._last_action)        
        new_params = self._estimator.get_params() + self._learning_rate * (reward + self._discount_factor * max_q - old_q) * grad
        self._estimator.set_params(new_params)        