import numpy as np
import matplotlib.pyplot as plt
from stock import *
from exchange import *
from qlearner import *
from environment import *

def graph_performance(wealths, figname):    
    plt.figure()
    plt.plot(range(len(wealths)), wealths)
    plt.title('Agent performance in {0} steps after training'.format(len(wealths)))
    plt.xlabel('iterations')
    plt.ylabel('cumulative wealth')
    plt.savefig(figname)
    plt.show()

def run_qmatrix_stock_trading():
    stock = OUStock(price=110, kappa=0.1, mu=150, sigma=0.2, tick=0.1, band=1000)
    lot = 100
    actions = list(range(-3*lot, 4*lot, lot))
    learner = QMatrix(actions, epsilon=0.1, learning_rate=0.5, discount_factor=0.999)
    exchange = StockExchange(stock, lot=lot, impact=0, max_holding=1000)
    environment = StockTradingEnvironment(stock, learner, exchange)
    
    # for initial training and burn in
    environment.run(util=1e-3, nrun=int(1e6))    
    # for graphing pnl after training, run again the above 5k times
    wealths = environment.run(util=1e-3, nrun=5000, report=True)
    graph_performance(wealths, 'qmatrix_performance')
   
if __name__ == '__main__':
    run_qmatrix_stock_trading()    