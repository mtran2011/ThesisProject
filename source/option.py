from math import log, exp
from scipy.stats import norm
from stock import GBMStock

class EuropeanOption(object):
    ''' European stock option
    Attributes:
        stock (GBMStock): the underlying stock object
        strike (float): the strike
        expiry (int): the original time to expiry        
        rate (float): risk free rate
        is_call (boolean): True if this is a call
        _tau (int): the remaining time to expiry
        _price (float): the price of this option
        _delta (float): the delta of this option
    '''
    def __init__(self, stock : GBMStock, strike : float, expiry : int, is_call=True):
        if expiry <= 0:
            raise ValueError('original expiry must be positive')
        self.stock = stock
        self.strike = strike
        self.expiry = expiry        
        self.rate = stock.mu
        self.is_call = is_call
        self._tau = expiry
        self._price = None
        self._delta = None
        self.update_price()
    
    def update_price(self):
        ''' Update self._price and self._delta. These 2 data must be private.
        The only way to get option price and delta is via update_price() 
        This is to ensure option price is always in line with stock price
        Returns:
            float: the option price rounded to tick
            float: the option delta
        '''
        tick = self.stock.tick
        r, k, T = self.rate, self.strike, self._tau

        if T < 0:
            self._price = 0
            self._delta = 0
            return self._price, self._delta
        s = self.stock.get_price()
        if T == 0:
            self._price = max(s-k, 0) if self.is_call else max(k-s, 0)

            if self.is_call:
                self._delta = 1 if s > k else 0
            else:
                self._delta = -1 if s < k else 0
            return self._price
        
        sig = self.stock.sigma        

        d1 = (log(s / k) + (r + 0.5 * sig**2) * T) / (sig * T**0.5)
        d2 = d1 - sig * T**0.5
        if self.is_call:
            self._delta = norm.cdf(d1)
            self._price = s * self._delta - k * exp(-r*T) * norm.cdf(d2)
        else:
            self._delta = -norm.cdf(-d1)
            self._price = s * self._delta + k * exp(-r*T) * norm.cdf(-d2)
        return self._price