from math import log, exp
from scipy.stats import norm

class EuropeanStockOption(object):
    ''' European stock option
    Attributes: 
        stock (Stock): the underlying stock object
        strike (float): the strike
        expiry (int): the original time to expiry
        tau (int): the remaining time to expiry
        rate (float): risk free rate
        is_call (boolean): True if this is a call
        _price (float): the price of this option
        _delta (float): the delta of this option
    '''
    def __init__(self, stock, strike, expiry, rate, is_call=True):
        self.stock = stock
        self.strike = strike
        self.expiry = expiry
        self.tau = expiry
        self.rate = rate
        self.is_call = is_call
        self._price = None
        self._delta = None
        self.update_price()
    
    def get_price(self):
        ''' Return option price rounded same as stock tick
        '''        
        return round(max(self._price, 0), {0.01: 2, 0.1: 1, 1: 0}[self.stock.tick])

    def get_delta(self):
        ''' Return delta of one option
        '''
        return self._delta

    def reset_tau(self):
        ''' Reset tau to original expiry. This is used when starting a new learning episode.
        '''
        self.tau = self.expiry
        self.update_price()

    def update_price(self):
        ''' Update self._price and self._delta
        '''
        r, k, T = self.rate, self.strike, self.tau        

        if T < 0:
            self._price = 0
            self._delta = 0            
            return self._price
        s = self.stock.get_price()
        if T == 0:
            self._price = max(s-k, 0) if self.is_call else max(-s+k, 0)
            if self.is_call:
                self._delta = 1 if s > k else 0
            else:
                self._delta = -1 if s < k else 0
            return self._price
        
        sig = self.stock.sigma        

        d1 = (log(s / k) + (r + 0.5 * sig**2) * T) / (sig * T**0.5)
        d2 = d1 - sig * T**0.5
        print(T)
        if self.is_call:
            self._delta = norm.cdf(d1)
            self._price = s * self._delta - k * exp(-r*T) * norm.cdf(d2)
        else:
            self._delta = -norm.cdf(-d1)
            self._price = s * self._delta + k * exp(-r*T) * norm.cdf(-d2)
        return self._price