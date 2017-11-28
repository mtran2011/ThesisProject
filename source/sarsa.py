''' Module for SARSA learner
'''

import abc
from learner import Learner, MatrixLearner

class SarsaLearner(Learner):
    ''' Abstract base class for a Sarsa learner (on policy)
    '''

    @abc.abstractmethod
    def _train_internally(self, reward, next_q):
        ''' Train internal model using reward and next_q:=Q(s',a')
        The internal training can be:
            for matrix: update Q(_last_state, _last_action) += learning_rate * (reward + gamma * next_q - old_q)
            for SemiGrad: via grad descent, update the parameters in the function estimator
        Args:
            reward (float): the reward seen after taking self._last_action
            next_q (float): value of Q(s',a') where s' is the new_state, a' is new action based on epsilon-greedy (on policy)
        Returns:
            None: train internal model based on (self._last_state, self._last_action, reward, new_state)
        '''
        raise NotImplementedError('Not implemented at SarsaLearner base class')

    # Override
    def learn(self, reward, new_state):
        action, next_q = self._find_action_greedily(new_state, use_epsilon=True, return_q=True)
        if self._last_action is not None and self._last_state is not None:
            self._train_internally(reward, next_q)
        self._last_action = action
        self._last_state = new_state
        return action

class SarsaMatrix(MatrixLearner, SarsaLearner):
    ''' Abstract class for a Sarsa-learner matrix that holds the values of Q(s,a)
    '''

    # Override
    def _train_internally(self, reward, next_q):
        # cannot train if never seen a state or took an action before
        if self._last_action is None or self._last_state is None:
            return None

        old_q = self._get_q(self._last_state, self._last_action)
        new_q = old_q + self._learning_rate * (reward + self._discount_factor * next_q - old_q)
        self._Q[(self._last_state, self._last_action)] = new_q

class TabularSarsaMatrix(SarsaMatrix):
    ''' The discrete, tabular Sarsa-matrix learner
    '''
    # Override
    def _get_q(self, state, action):
        return self._Q.get((state, action), 0)