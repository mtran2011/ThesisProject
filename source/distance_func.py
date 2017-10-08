import random
import numpy as np

def p_norm(x1, x2, p=2):
    '''
    Args:
        x1 (list): list of coordinates of the first point
        x2 (list): list of coordinates of the second point
    Returns:
        float: distance in L1 norm between x1 and x2
    '''
    x1 = np.array(x1)
    x2 = np.array(x2)
    return np.linalg.norm(x1 - x2, ord=p)




class QMatrixHeuristic(QMatrix):
    '''
    Attributes:        
        _dist_func (function): to calculate distance between two points (s1, a1) and (s2, a2)
        sample_size (int): how many samples to take when heuristically averaging and estimating 
    '''

    def __init__(self, actions, dist_func, epsilon=0.1, learning_rate=0.1, discount_factor=0.999):
        super().__init__(actions, epsilon, learning_rate, discount_factor)        
        self._dist_func = dist_func
        self.sample_size = 100
    
    def _get_q(self, state, action):
        if (state, action) in self._Q:
            return self._Q[(state, action)]
        else:
            # guess the value of Q(state, action) via sampling and averaging
            sample_size = min(self.sample_size, len(self._Q))
            batch = random.sample(self._Q.keys(), sample_size)
            distances, q_vals = [], []
            x1 = [*state, action] # coordinate of this point
            for that_state, that_action in batch:
                x2 = [*that_state, that_action] # coordinate of that point
                distance = self._dist_func(x1, x2)
                q_val = self._Q[(that_state, that_action)]
                distances.append(distance)
                q_vals.append(q_val)
            
            distances = np.array(distances)
            q_vals = np.array(q_vals)
            estimate = distances.dot(q_vals) / distances.sum()
            self._Q[(state, action)] = estimate
            return estimate