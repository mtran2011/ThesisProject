import abc
import random

class Stock(object):
    ''' Abstract base class for a stock object
    Attributes:
        spot (float): the current spot price, must be rounded to 2 decimal points (cents)
    '''
    
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, spot):
        if not spot:
            raise ValueError('spot price input cannot be None')
        self.spot = round(spot, 2)
        
    def simulate_price(self, mu, sigma, dt):
        ''' Use dS = mu * dt + sigma * sqrt(dt) * standard normal N(0,1)
        Based on dS = mu(.) * dt + sigma(.) * dW where var(dW) = dt        
        '''
        new_spot = self.spot + mu * dt + sigma * (dt**0.5) * random.gauss(0.0, 1.0)
        self.spot = round(new_spot, 2)
        return self.spot