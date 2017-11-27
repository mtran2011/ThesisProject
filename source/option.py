from math import log, exp
from scipy.stats import norm
from stock import GBMStock

class EuropeanOption(object):
    ''' European stock option on a non-dividend paying stock
    Attributes:
        _stock (GBMStock): the underlying stock object
        _strike (float): the strike
        _expiry (int): the original time to expiry        
        _is_call (bool): True if this is a call
        _rate (float): risk free rate
        _tau (int): the remaining time to expiry
        _price (float): the price of this option
        _delta (float): the delta of this option
    '''
    def __init__(self, stock : GBMStock, strike : float, expiry : int, is_call : bool):
        if expiry <= 0:
            raise ValueError('original expiry must be positive')
        self.stock = stock
        self.strike = strike
        self.expiry = expiry
        self.is_call = is_call
        self.tau = expiry
        
        self._rate = stock.mu        
        self._price = None
        self._delta = None
        self.update_price()
    
    def has_expired(self):
        ''' Return True if this option has expired
        '''
        return self.tau < 0

    def update_price(self):
        ''' Update self._price and self._delta. These 2 data must be private.
        The only way to get option price and delta is via update_price() 
        This is to ensure option price is always in line with stock price
        Returns:
            float: the option price rounded to tick
            float: the option delta
        '''
        r, k, T = self._rate, self.strike, self.tau

        if T < 0:
            self._price = 0
            self._delta = 0
            return self._price, self._delta
        
        s = self.stock.get_price()
        if T == 0:
            self._price = max(s-k, 0) if self.is_call else max(k-s, 0)
            self._price = round(self._price, {0.01: 2, 0.1: 1, 1: 0}[self.stock.tick])
            if self.is_call:
                self._delta = 1 if s > k else 0
            else:
                self._delta = -1 if s < k else 0
            return self._price, self._delta
        
        sig = self.stock.sigma
        d1 = (log(s / k) + (r + 0.5 * sig**2) * T) / (sig * T**0.5)
        d2 = d1 - sig * T**0.5
        if self.is_call:
            self._delta = norm.cdf(d1)
            self._price = s * self._delta - k * exp(-r*T) * norm.cdf(d2)            
        else:
            self._delta = -norm.cdf(-d1)
            self._price = s * self._delta + k * exp(-r*T) * norm.cdf(-d2)            
        self._price = round(self._price, {0.01: 2, 0.1: 1, 1: 0}[self.stock.tick])
        return self._price, self._delta

class Pair(object):
    ''' To wrap an option and the underlying stock 
    Memorize their attributes and report without doing the expensive repricing function
    '''
    def __init__(self, stock : GBMStock, strike : float, expiry : int, is_call : bool):
        # to memorize the original attributes of stock and option
        self._strike = strike
        self._expiry = expiry
        self._is_call = is_call        
        self._original_stock_price = stock.get_price()

        self._stock = stock
        self._option = EuropeanOption(self._stock, self._strike, self._expiry, self._is_call)
        
        self._option_price, self._option_delta = self._option.update_price()
    
    def get_stock(self):
        ''' Todo: should we expose the stock like this
        '''
        return self._stock
    
    def reset_episode(self):
        ''' Reset option to a new option with the original expiry and reset stock to original price
        '''        
        self._stock.set_price(self._original_stock_price)
        self._option = EuropeanOption(self._stock, self._strike, self._expiry, self._is_call)
        self._option_price, self._option_delta = self._option.update_price()

    def get_option_price(self):
        ''' To report current option price
        '''
        return self._option_price
    
    def get_option_delta(self):
        ''' To report current option delta
        '''
        return self._option_delta
    
    def check_option_expired(self):
        ''' Return True if this option has expired
        '''
        return self._option.has_expired()
    
    def simulate_stock_price(self, dt=1):
        ''' Simulate the stock for one step dt, decrement option.tau, and reprice the option
        Returns:
            float: new_stock_price 
            float: new_option_price
        '''
        self._stock.simulate_price(dt)
        pass