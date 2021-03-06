''' Module for SARSA learner
'''

import abc
import random
from math import log2
import numpy as np
from learner import Learner, MatrixLearner
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

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

        self._count += 1
        self._epsilon = min(1 / log2(self._count), self._epsilon)

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

class KernelSmoothingSarsaMatrix(SarsaMatrix):
    ''' Use kernel smoothing or local regression to estimate Q(s,a) if this has not been found before
    Attributes:
        regressor (KernelRegressor): a regressor that can fit(X,Y) and predict(X) using kernel smoothing method
        sample_size (int): how many samples to take from existing Q(s,a) as training data for kernel smoother
    '''

    def __init__(self, actions, regressor, epsilon, learning_rate, discount_factor, sample_size=30):
        super().__init__(actions, epsilon, learning_rate, discount_factor)
        self.regressor = regressor
        self.sample_size = sample_size
    
    # Override
    def _get_q(self, state, action):
        if not self._Q:
            return 0
        if (state, action) in self._Q:
            return self._Q[(state, action)]
        
        # now that (state, action) is not in Q, try to estimate Q(state, action) via smoothing
        # first sample a training data from existing Q(s,a)
        sample_size = min(self.sample_size, len(self._Q))
        # batch is a list of tuple (state, action)
        batch = random.sample(self._Q.keys(), sample_size)
        X = [[*s, a] for s, a in batch]
        X = np.array(X)
        Y = np.array([self._Q[key] for key in batch]).reshape(len(batch),1)        
        self.regressor.fit(X,Y)

        # estimate must be a float, scalar
        x = np.array([*state, action]).reshape(1, len(state)+1)
        estimate = np.asscalar(self.regressor.predict(x))
        self._Q[(state, action)] = estimate
        return estimate

class RandomForestSarsaMatrixVersion1(SarsaMatrix):
    ''' Use random forest on a random sample of existing values of Q(s,a) to estimate new Q(s,a)
    Each time you see a new state, estimate Q(s,a) for all (s,a) pairs with that state
    The random forest is first fit on a sample of existing Q, then used to predict new Q
    Attributes:
        sample_size (int): how many samples to take from existing Q(s,a) as training data
    '''
    def __init__(self, actions, epsilon, learning_rate, discount_factor, max_nfeatures=2):
        super().__init__(actions, epsilon, learning_rate, discount_factor)
        self.rf = RandomForestRegressor(
            n_estimators=10, max_features=max_nfeatures,
            min_samples_leaf=5, n_jobs=2)
        self.sample_size = 100
    
    # Override
    def _get_q(self, state, action):
        if not self._Q:
            return 0
        if (state, action) in self._Q:
            return self._Q[(state, action)]
        
        # prepare training data 
        self.sample_size = max(self.sample_size, len(self._Q) // 10)
        sample_size = min(self.sample_size, len(self._Q))
        batch = random.sample(self._Q.keys(), sample_size)
        X = [[*s, a] for s, a in batch]
        X = np.array(X)
        Y = np.array([self._Q[key] for key in batch])
        self.rf.fit(X,Y)

        # now predict on this new (s,a)
        # important note: predict on (state,a) for all a where (state,a) not yet in Q        
        for a in self._actions:
            if (state, a) not in self._Q:
                x = np.array([*state, a]).reshape(1, len(state)+1)
                estimated_val = np.asscalar(self.rf.predict(x))
                self._Q[(state, a)] = estimated_val
                if a == action:
                    estimate = estimated_val
        
        return estimate

class RandomForestSarsaMatrixVersion2(SarsaMatrix):
    ''' Use random forest on all existing values of Q(s,a) to estimate new Q(s,a)
    The change here is that the forest is refitted once every 500 steps to all of existing Q
    '''
    def __init__(self, actions, epsilon, learning_rate, discount_factor, max_nfeatures=2):
        super().__init__(actions, epsilon, learning_rate, discount_factor)
        self.rf = RandomForestRegressor(
            n_estimators=30, max_features=max_nfeatures,
            min_samples_leaf=5, n_jobs=2)
    
    # Override
    def _get_q(self, state, action):
        # in the first 500 training steps, do not estimate, just default to 0
        if not self._Q:
            return 0
        if (state, action) in self._Q:
            return self._Q[(state, action)]
        if self._count - 2 < 500:
            return 0
        
        # refit the random forest once every 500 steps
        if (self._count - 2) % 500 == 0:
            # prepare training data
            X, Y = [], []
            for key, value in self._Q.items():
                # key is a tuple of (s,a)
                X.append([*key[0], key[1]])
                Y.append(value)
            X = np.array(X)
            Y = np.array(Y)
            self.rf.fit(X, Y)
        # use the random forest to predict Q for this (state, action)
        x = np.array([*state, action]).reshape(1, len(state)+1)
        return np.asscalar(self.rf.predict(x))

class TreeSarsaMatrix(SarsaMatrix):
    ''' Use a single tree on all existing values of Q(s,a) to estimate new Q(s,a)
    The tree is refitted once every 150 learning steps
    '''
    def __init__(self, actions, epsilon, learning_rate, discount_factor):
        super().__init__(actions, epsilon, learning_rate, discount_factor)
        self.tree = DecisionTreeRegressor(criterion="mae", min_samples_leaf=5)
    
    # Override
    def _get_q(self, state, action):
        # in the first 150 training steps, do not estimate, just default to 0
        if not self._Q:
            return 0
        if (state, action) in self._Q:
            return self._Q[(state, action)]
        if self._count - 2 < 150:
            return 0

        # refit the tree once every 150 steps
        if (self._count - 2) % 150 == 0:
            # prepare training data
            X, Y = [], []
            for key, value in self._Q.items():
                # key is a tuple of (s,a)
                X.append([*key[0], key[1]])
                Y.append(value)
            X = np.array(X)
            Y = np.array(Y)
            self.tree.fit(X,Y)
        # use the tree to predict Q for this (state, action)
        x = np.array([*state, action]).reshape(1, len(state)+1)
        return np.asscalar(self.tree.predict(x))