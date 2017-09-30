import abc
import random

'''
Based on github.com/vmayoral/basic_reinforcement_learning/blob/master/tutorial1/qlearn.py
'''

class RLAgent(object):
    ''' Abstract base class for a reinforcement learning agent
    Attributes:
        _actions (list): the list of all possible actions it can take
        _last_action (object): the immediate previous action it took
        _last_state (object): to memorize the immediate previous state, for which it took _last_action
    '''
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, actions):
        if not actions:
            raise ValueError('list of possible actions cannot be empty')
                
        self._actions = actions
        self._last_action = None
        self._last_state = None        
    
    @abc.abstractmethod    
    def learn(self, reward, new_state):
        ''' Get a reward and see a new_state. Use this to update Q(_last_state, _last_action). 
        Find the best_action based on new_state
        Update _last_state <- new_state
        Update _last_action <- best_action        
        Args:
            reward (float): the reward seen after the previous action
            new_state (object): the new_state seen after the previous action
        Returns:
            best_action (object): take a new action 
        '''
        
class QLearner(RLAgent):
    ''' Class for a Q-learner that holds the values of Q(s,a)
    Attributes:
        _Q (dict): dict of key tuple (s,a) to float value Q(s,a)        
        _epsilon (float): constant in epsilon-greedy policy
        _learning_rate (float): the constant learning_rate
        _discount_factor (float): the constant discount_factor of future rewards        
    '''
    
    def __init__(self, actions, epsilon=0.1, learning_rate=0.001, discount_factor=0.999):        
        super().__init__(actions)
        self._Q = dict()
        self._epsilon = epsilon
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
    
    def _get_q(self, state, action):
        ''' Return existing Q(s,a) for a given state s and action a
        Args:
            state (object): state s
            action (object): action a
        Returns:
            float: the value of Q(s,a). Zero if the Q value for tuple (s,a) has never been assigned
        '''
        return self._Q.get((state, action), 0)
    
    def _find_action_greedily(self, state, use_epsilon=True, return_q=False):
        ''' Given the state, find the best action using epsilon-greedy
        Args:
            state (object): the given state
            use_epsilon (bool): True if the randomization using epsilon is to be used
            return_q (bool): True to return the found value of Q(state, a)
        Returns: 
            object: the action found by epsilon-greedy
            float: the value q(s,a) for state s found by epsilon-greedy
        '''        
        if random.random() < self._epsilon and use_epsilon:
            # with probability epsilon, choose from all actions with equal chance
            best_action = random.choice(self._actions)
            max_q = self._get_q(state, best_action)
        else:
            # choose a = arg max {action} of Q(state, action)
            q_values = [self._get_q(state, a) for a in self._actions]
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
            state (object): the state this agent saw
            action (object): the action taken by the agent
            reward (float): the reward seen after taking action
            new_state (object): the new_state seen after taking action
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
    def learn(self, reward, new_state):
        if not self._last_action or not self._last_state:
            action = self._find_action_greedily(new_state)
            self._last_action = action          
            self._last_state = new_state
            return action
        else:
            self._update_q(self._last_state, self._last_action, reward, new_state)
            action = self._find_action_greedily(new_state)
            self._last_action = action          
            self._last_state = new_state
            return action 