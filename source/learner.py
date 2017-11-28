import abc

class Learner(abc.ABC):
    ''' Abstract base class for a learning agent, either Q-learning or Sarsa
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
        With probability of epsilon, pick a random action. Else pick a greedy action.
        Args:
            state (tuple): the given state which is a tuple of state attributes
            use_epsilon (bool): True if the randomization using epsilon is to be used
            return_q (bool): True if user wants to return the found value of Q(state, a)
        Returns: 
            object: the action found by epsilon-greedy
            float: the value Q(s,a) for state s found by epsilon-greedy
        '''
        pass
    
    @abc.abstractmethod
    def _train_internally(self, reward, new_state):
        ''' Use reward and new_state to train the internal model.
        The internal training can be:
            for Q matrix: update Q(_last_state, _last_action) += (reward + gamma * maxQ(new_state,action)) * learning_rate
            for DQN: add the experience (s,a,r,s') to memory and train the internal neural network
            for SemiGradQLearner: via grad descent, update the parameters in the function estimator
        Args:
            reward (float): the reward seen after taking self._last_action
            new_state (tuple): the new_state seen after taking self._last_action
        Returns:
            None: train internal model based on (self._last_state, self._last_action, reward, new_state)
        '''
        pass

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
        pass

    def reset_last_action(self):
        ''' Reset to prepare to play a new episode
        '''
        self._last_action = None
        self._last_state = None