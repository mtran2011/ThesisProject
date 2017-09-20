import abc
import random

'''
Based on github.com/vmayoral/basic_reinforcement_learning/blob/master/tutorial1/qlearn.py
'''

class RLAgent(object):
    ''' Abstract base class for a reinforcement learning agent
    Attributes:
        actions (list-like): the list of all possible actions it can take
        last_action (object): the immediate previous action it took
        current_state (object): to memorize the current_state it is observing
    '''
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, actions, current_state):
        if not actions:
            raise ValueError('list of possible actions cannot be empty')
        if not current_state:
            raise ValueError('current_state the agent is seeing cannot be None')
        
        self.actions = actions
        self.last_action = None
        self.current_state = current_state
    
    @abc.abstractmethod
    def take_action(self):
        ''' Deliver an action based on current_state it is observing
        Update self.last_action to the one just taken
        Returns:
            action (object): take an action based on current_state it is observing
        '''
    
    @abc.abstractmethod    
    def learn(self, reward, new_state):
        ''' Get a reward and see a new_state. Use this to learn. 
        Update current_state attribute to new_state. 
        Then take_action based on new current_state
        Args:
            reward (float): the reward seen after the previous action
            new_state (object): the new_state seen after the previous action
        Returns:
            action (object): take a new action 
        '''

class QLearner(RLAgent):
    ''' Class for a Q-learner that holds the values of Q(s,a)
    Attributes:
        Q (dict): dict of key tuple (s,a) to float value Q(s,a)        
        epsilon (float): constant in epsilon-greedy policy
        learning_rate (float): the constant learning_rate
        discount_factor (float): the constant discount_factor of future rewards        
    '''
    
    def __init__(self, actions, current_state, epsilon=0.1, learning_rate=0.1, discount_factor=0.9999):
        super().__init__(actions, current_state)
        
        self.Q = dict()        
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
    
    def get_q(self, state, action):
        ''' Return existing Q(s,a) for a given state s and action a
        Args:
            state (object): state s
            action (object): action a
        Returns:
            float: the value of Q(s,a). Zero if the value for tuple (s,a) has never been assigned
        '''
        return self.Q.get((state, action), 0)
    
    def find_action_greedily(self, state, use_epsilon=True, return_q=False):
        ''' Given the current state, find the best action using epsilon-greedy
        Args:
            state (object): the current state
            use_epsilon (bool): True if the randomization using epsilon is to be used
            return_q (bool): True to return the greedy value of Q(state, a)
        Returns: 
            object: the action found by epsilon-greedy
            float: the value q(s,a) for current state found by epsilon-greedy
        '''        
        if random.random() < self.epsilon and use_epsilon:
            # with probability epsilon, choose from all actions with equal chance
            best_action = random.choice(self.actions)
            max_q = self.get_q(state, best_action)
        else:
            # choose a = arg max {action} of Q(state, action)
            best_action = self.actions[0]
            max_q = self.get_q(state, best_action)
            for action in self.actions:
                q = self.get_q(state, action)
                if q > max_q: 
                    max_q = q
                    best_action = action
        
        if return_q:
            return best_action, max_q
        else:
            return best_action                
    
    def update_q(self, current_state, action, reward, new_state):
        ''' Update Q(current_state, action) after observing reward and new_state
        From the current_state, agent takes action a, then observing reward and new_state
        Thus use the reward and max over {a} of Q(new_state, a) to update Q(current_state, action)
        Args:
            current_state (object): the current_state this agent is seeing
            action (object): the action taken by the agent
            reward (float): the reward seen after taking action
            new_state (object): the new_state seen after taking action
        Returns:
            float: the update value of Q(current_state, action)
        '''
        old_q = self.get_q(current_state, action)
        # find the max over all {a} of Q(new_state, a)
        _, max_q = find_action_greedily(new_state, use_epsilon=False, return_q=True)
        new_q = old_q + self.learning_rate * (reward + self.discount_factor * max_q - old_q)
        self.Q[(current_state, action)] = new_q
        return new_q
    
    # Override base class abstractmethod
    def take_action(self):
        ''' Deliver an action based on current_state it is observing
        Use epsilon-greedy to take an action
        Returns:
            action (object): take an action based on current_state it is observing
        '''        
        action = self.find_action_greedily(self.current_state)
        self.last_action = action
        return action
    
    # Override base class abstractmethod
    def learn(self, reward, new_state):
        ''' Get a reward and see a new_state. Use this to learn. 
        Update current_state attribute to new_state. 
        Then take_action based on new current_state
        Args:
            reward (float): the reward seen after the previous action
            new_state (object): the new_state seen after the previous action
        Returns:
            action (object): take a new action 
        '''
        if not self.last_action:
            self.current_state = new_state
            return self.take_action()
        else:
            self.update_q(self.current_state, self.last_action, reward, new_state)
            self.current_state = new_state
            return self.take_action()    