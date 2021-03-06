import abc
import random
from math import exp, log

class Stock(abc.ABC):
    ''' Abstract base class for a stock object
    Attributes:
        _price (float): the current spot price in USD, Must Be Rounded to 1 decimal point        
        _max (float): the max price this stock can reach
        _min (float): the lowest price this stock can reach, subject to 0
        tick (float): if the stock price changes, it must change in multiple of ticks
        band (int): the max and min price is initial price +/- (band * tick)
    '''
    
    def __init__(self, price, tick=0.1, band=1000):
        if tick not in [0.01, 0.1, 1]: 
            raise ValueError('tick can only be 0.01 or 0.1 or 1')
            
        self._price = round(max(0,price), {0.01: 2, 0.1: 1, 1: 0}[tick])
        self._max = self._price + tick * band
        self._min = max(0, self._price - tick * band)
        self.tick = tick        
        self.band = band

    def get_price(self):
        '''
        Returns:
            float: the current _price
        '''
        return self._price

    def set_price(self, price):
        ''' 
        Args:
            price (float): to set price
        '''        
        self._price = round(max(0,price), {0.01: 2, 0.1: 1, 1: 0}[self.tick])
        self._price = min(self._price, self._max)
        self._price = max(self._price, self._min)

    @abc.abstractmethod
    def simulate_price(self, dt=1.0):
        ''' Simulate and update self._price over time step dt based on some internal models
        Args:
            dt (float): length of time step
        Returns:
            float: the new updated price
        '''
        pass

class OUStock(Stock):
    ''' Stock with dS following an OU process
    dS = kappa * (mu - S) * dt + sigma * dW where var(dW) = dt
    Attributes:
        kappa (float): mean reversion speed
        mu (float): mean reversion level
        sigma (float): volatility
    '''
    
    def __init__(self, price, kappa, mu, sigma, tick=0.1, band=1000):
        super().__init__(price, tick, band)
        self.kappa = kappa
        self.mu = mu
        self.sigma = sigma
    
    # Override base class abstractmethod
    def simulate_price(self, dt=1.0):
        dW = dt**0.5 * random.gauss(0.0, 1.0)
        new_price = self._price + self.kappa * (self.mu - self._price) * dt + self.sigma * dW
        self.set_price(new_price)        
        return self._price

class OULogStock(OUStock):
    ''' Stock with dS following an OU process
    dlogS = kappa * (mu - logS) * dt + sigma * dW where var(dW) = dt    
    '''

    # Override base class abstractmethod
    def simulate_price(self, dt=1.0):
        dW = dt**0.5 * random.gauss(0.0, 1.0)
        old_log = log(max(1e-4, self._price))
        new_log = old_log + self.kappa * (self.mu - old_log) * dt + self.sigma * dW
        self.set_price(exp(new_log))
        return self._price

class GBMStock(Stock):
    ''' Geometric Brownian motion stock
    dlogS = (mu - 0.5 * sigma**2) * dt + sigma * dW
    Attributes:
        mu (float): the drift 
        sigma (float): volatility
    '''
    def __init__(self, price, mu, sigma, tick=0.1, band=1000):
        super().__init__(price, tick, band)        
        self.mu = mu
        self.sigma = sigma
    
    # Override base class abstractmethod
    def simulate_price(self, dt=1.0):
        dW = dt**0.5 * random.gauss(0.0, 1.0)
        dlogS = (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * dW
        new_price = self._price * exp(dlogS)
        self.set_price(new_price)        
        return self._price