import abc
import random
import math
import numpy as np

'''
Based on github.com/vmayoral/basic_reinforcement_learning/blob/master/tutorial1/qlearn.py
'''

class QLearner(metaclass=abc.ABCMeta):
    ''' Abstract base class for a reinforcement learning agent
    Attributes:
        _actions (list): the list of all possible actions it can take
        _last_action (object): the immediate previous action it took
        _last_state (tuple): to memorize the immediate previous state, for which it took _last_action
    '''    
    
    def __init__(self, actions):
        if not actions or not isinstance(actions, list):
            raise ValueError('actions cannot be empty and must be a list')                
        self._actions = actions
        self._last_action = None
        self._last_state = None        
    
    @abc.abstractmethod
    def _find_action_greedily(self, state, use_epsilon=True, return_q=False):
        ''' Given the state, find the best action using epsilon-greedy
        Args:
            state (tuple): the given state which is a tuple of state attributes
            use_epsilon (bool): True if the randomization using epsilon is to be used
            return_q (bool): True to return the found value of Q(state, a)
        Returns: 
            object: the action found by epsilon-greedy
            float: the value q(s,a) for state s found by epsilon-greedy
        '''
        pass
    
    @abc.abstractmethod
    def _train_internally(self, reward, new_state):
        ''' Use reward and new_state to train the internal model.
        The internal training can be:
            for discrete Q matrix: update Q(_last_state, _last_action)
            for DQN: add the experience (s,a,r,s') to memory and train the internal neural network
            for SemiGradQLearner: via grad descent, update the parameters in the function estimator
        Args:
            reward (float): the reward seen after taking self._last_action
            new_state (tuple): the new_state seen after taking self._last_action
        Returns:
            None: train internal model based on (self._last_state, self._last_action, reward, new_state)
        '''
        pass

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
        # if this agent has never taken any action before
        if not self._last_action or not self._last_state:
            action = self._find_action_greedily(new_state)
            self._last_action = action          
            self._last_state = new_state
            return action
        else:            
            self._train_internally(reward, new_state)
            action = self._find_action_greedily(new_state)
            self._last_action = action          
            self._last_state = new_state
            return action
        
class QMatrix(QLearner):
    ''' Class for a Q-learner that holds the values of Q(s,a)
    Attributes:
        _Q (dict): dict of key tuple (s,a) to float value Q(s,a)        
        _epsilon (float): constant in epsilon-greedy policy
        _learning_rate (float): the constant learning_rate
        _discount_factor (float): the constant discount_factor of future rewards        
    '''
    
    def __init__(self, actions, epsilon=0.1, learning_rate=0.5, discount_factor=0.999):        
        super().__init__(actions)
        self._Q = dict()
        self._epsilon = epsilon
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
    
    def _get_q(self, state, action):
        ''' Return existing Q(s,a) for a given state s and action a
        Args:
            state (tuple): state s, a tuple of state attributes
            action (object): action a
        Returns:
            float: the value of Q(s,a). Zero if the Q value for tuple (s,a) has never been assigned
        '''
        return self._Q.get((state, action), 0)
        
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
    
    def _update_q(self, state, action, reward, new_state):
        ''' Update Q(state, action) after observing reward and new_state
        From the state, agent takes action, then observing reward and new_state
        Thus use the reward and max over {a} of Q(new_state, a) to update Q(state, action)
        Args:
            state (tuple): the state this agent saw
            action (object): the action taken by the agent
            reward (float): the reward seen after taking action
            new_state (tuple): the new_state seen after taking action
        Returns:
            float: the updated value of Q(state, action)
        '''
        old_q = self._get_q(state, action)
        # find the max over all {a} of Q(new_state, a)
        _, max_q = self._find_action_greedily(new_state, use_epsilon=False, return_q=True)
        new_q = old_q + self._learning_rate * (reward + self._discount_factor * max_q - old_q)
        self._Q[(state, action)] = new_q
        return new_q    
    
    # Override base class abstractmethod
    def _train_internally(self, reward, new_state):
        self._update_q(self._last_state, self._last_action, reward, new_state)
        return None

class QMatrixHeuristic(QMatrix):
    '''
    Attributes:        
        dist_func (function): to calculate distance between two points (s1, a1) and (s2, a2)
        sample_size (int): how many samples to take when heuristically averaging and estimating Q(s,a)
    '''

    def __init__(self, actions, dist_func, epsilon=0.1, learning_rate=0.5, discount_factor=0.999):
        super().__init__(actions, epsilon, learning_rate, discount_factor)        
        self.dist_func = dist_func
        self.sample_size = 100
    
    def _get_q(self, state, action):
        if (state, action) in self._Q:
            return self._Q[(state, action)]
        elif not self._Q:
            return 0
        else:
            # guess the value of Q(state, action) via sampling and averaging
            sample_size = min(self.sample_size, len(self._Q))
            batch = random.sample(self._Q.keys(), sample_size)
            distances, q_vals = [], []
            x1 = [*state, action] # coordinate of this point
            for that_state, that_action in batch:
                x2 = [*that_state, that_action] # coordinate of that point
                distance = self.dist_func(x1, x2)
                q_val = self._Q[(that_state, that_action)]
                distances.append(distance)
                q_vals.append(q_val)
            
            distances = np.array(distances)
            q_vals = np.array(q_vals)
            estimate = distances.dot(q_vals) / distances.sum()
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
    
    def __init__(self, actions, model, epsilon=0.1, discount_factor=0.999):        
        super().__init__(actions)
        self._epsilon = epsilon        
        self._discount_factor = discount_factor
        self._memory = []
        self._model = model # todo
        
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
        # memorize this experience and then train the neural network
        self._memory.append((self._last_state, self._last_action, reward, new_state))
        self._replay()
        return None

class SemiGradQLearner(QLearner):
    ''' Class for a Q-learner that uses a parametric function approximator to estimate Q(s,a) for all a
    Attributes:
        _epsilon (float): constant in epsilon-greedy policy
        _learning_rate (float): the constant learning_rate
        _discount_factor (float): the constant discount_factor of future rewards        
        _estimator (QFunctionEstimator): a parametric function estimator
    '''
    
    def __init__(self, actions, estimator, epsilon=0.1, learning_rate=0.5, discount_factor=0.999):
        super().__init__(actions)
        self._epsilon = epsilon
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._estimator = estimator
    
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
        old_q = self._estimator.estimate_q(self._last_state, self._last_action)
        _, max_q = self._find_action_greedily(new_state, use_epsilon=False, return_q=True)
        # gradient with respect to the parameters
        grad = self._estimator.eval_gradient(self._last_state, self._last_action)        
        new_params = self._estimator.get_params() + self._learning_rate * (reward + self._discount_factor * max_q - old_q) * grad
        self._estimator.set_params(new_params)
        return None