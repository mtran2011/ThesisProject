import abc
import random

class Stock(object):
    ''' Abstract base class for a stock object
    Attributes:
        _price (float): the current spot price in USD, Must Be Rounded to 2 decimal points
    '''
    
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, price):
        if not price:
            raise ValueError('price input cannot be None')
        self._price = round(max(0,price), 2)
    
	def get_price(self):
		return self._price
	
    def set_price(self, price):
        ''' 
        Args:
            price (float): to set price
        '''
        if not price:
            raise ValueError('price input cannot be None')
        self._price = round(max(0,price), 2)
    
    @abc.abstractmethod
    def simulate_price(self, dt):
        ''' Simulate and update self._price over time step dt based on some internal models
        Args:
            dt (float): length of time step
        Returns:
            float: the new updated price
        '''

class OUStock(Stock):
    ''' Stock with dS following an OU process
    dS = kappa * (mu - S) * dt + sigma * dW where var(dW) = dt
    Attributes:
        kappa (float): mean reversion speed
        mu (float): mean reversion level
        sigma (float): volatility
    '''
    
    def __init__(self, price, kappa, mu, sigma):
        super().__init__(price)
        self.kappa = kappa
        self.mu = mu
        self.sigma = sigma
    
    # Override base class abstractmethod    
    def simulate_price(self, dt):
        dW = dt**0.5 * random.gauss(0.0, 1.0)
        new_price = self._price + self.kappa * (self.mu - self._price) * dt + self.sigma * dW
        self._price = round(max(0,new_price), 2)
        return self._price    