import abc
from qlearner import RLAgent
from exchange import StockExchange

class StockTradingEnvironment(object):
    ''' Reinforcement learning environment for single stock trading by an agent
    Attributes:
        agent (RLAgent): 
        exchange (StockExchange): 
    '''