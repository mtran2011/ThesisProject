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