import numpy as np

def inverse_norm_p(x1, x2, p=2):
    '''
    Args:
        x1 (list): list of coordinates of the first point
        x2 (list): list of coordinates of the second point
    Returns:
        float: inverse of distance in L-p norm between x1 and x2
    '''
    x1 = np.array(x1)
    x2 = np.array(x2)
    return 1 / np.linalg.norm(x1 - x2, ord=p)