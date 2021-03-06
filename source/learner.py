''' Module for abstract base class for learner
'''

import abc
import random
import operator

class Learner(abc.ABC):
    ''' Abstract base class for a learning agent, either Q-learning or Sarsa
    Attributes:
        _actions (tuple): the list of all possible actions it can take
        _epsilon (float): constant used in epsilon greedy
        _count (int): number of learning steps it has done
        _last_action (object): the immediate previous action it took
        _last_state (tuple): to memorize the immediate previous state, for which it took _last_action
    '''
    def __init__(self, actions, epsilon):
        if not actions or not isinstance(actions, tuple):
            raise ValueError('actions cannot be empty and must be a tuple')                
        self._actions = actions
        self._epsilon = epsilon
        self._count = 2 # to avoid divide by zero in log2(count)
        self._last_action = None
        self._last_state = None
    
    @abc.abstractmethod
    def _find_action_greedily(self, state, use_epsilon=True, return_q=False):
        ''' Given the state, find the best action using epsilon-greedy
        With probability of epsilon, pick a random action. Else pick a greedy action.
        Args:
            state (tuple): the given state which is a tuple of state attributes
            use_epsilon (bool): True if the randomization using epsilon is to be used
            return_q (bool): True if user wants to return the found value of Q(state, a)
        Returns: 
            object: the action found by epsilon-greedy
            float: the value Q(s,a) for state s found by epsilon-greedy
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def learn(self, reward, new_state):
        ''' Get a reward and see a new_state. Use these to do some internal training if _last_action is not None. Return a new action.        
        Update _last_state <- new_state
        Update _last_action <- best_action        
        Args:
            reward (float): the reward seen after the previous action
            new_state (tuple): the new_state seen after the previous action
        Returns:
            best_action (object): take a new action based on new_state
        '''
        raise NotImplementedError

    def reset_last_action(self):
        ''' Reset to prepare to play a new episode
        '''
        self._last_action = None
        self._last_state = None

class MatrixLearner(Learner):
    ''' Abstract base class for a Q-matrix or Sarsa-matrix
    Attributes:
        _Q (dict): dict of key tuple (s,a) to float value Q(s,a)
        _epsilon (float): constant in epsilon-greedy policy
        _learning_rate (float): the constant learning_rate
        _discount_factor (float): the constant discount_factor of future rewards
    '''

    def __init__(self, actions, epsilon, learning_rate, discount_factor):
        super().__init__(actions, epsilon)
        self._Q = dict()        
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
        raise NotImplementedError
        
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
            index, max_q = max(enumerate(q_values), key=operator.itemgetter(1))
            
            # max_q = 0 means this state has never been visited
            # if so take a random action
            if max_q == 0:
                best_action = random.choice(self._actions)
            else:
                best_action = self._actions[index]
        
        if return_q:
            return best_action, max_q
        else:
            return best_action