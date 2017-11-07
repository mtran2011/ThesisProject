import abc
from math import log, exp
from scipy.stats import norm

class EuropeanStockOption(object):
    ''' European stock option
    Attributes: 
        stock (Stock): the underlying stock object
        k (float): the strike
        tau (int): time to expiry (in days)
        r (float): risk free rate
        is_call (boolean): True if call
    '''
    def __init__(self, stock, k, tau, r=0, is_call=True):
        self.stock = stock
        self.k = k
        self.tau = tau
        self.r = r
        self.is_call = is_call      

    def find_price(self):
        if self.tau < 0:
            return 0        
        s = self.stock.get_price()
        if self.tau == 0:
            return max(s-self.k, 0) if self.is_call else max(-s+self.k, 0)
        
        sig = self.stock.sigma
        d1 = (log(s / self.k) + (r + 0.5 * sig**2) * self.tau) / (sig * self.tau**0.5)
        d2 = d1 - sig * self.tau**0.5
        if self.is_call:
            return s * norm.cdf(d1) - k * exp(-self.r*self.tau) * norm.cdf(d2)
        else:
            return -s * norm.cdf(-d1) + k * exp(-self.r*self.tau) * norm.cdf(-d2)