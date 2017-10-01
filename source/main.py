import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stock import Stock, OUStock
from exchange import StockExchange
from qlearner import QLearner, QMatrix

def run_stock_agent(stock, learner, exchange, util_const=1e-4, nrun=100000, report=False):
    ''' Run a stock trading agent for nrun iterations
    Args:
        stock (Stock): the stock this agent is trading
        learner (QLearner): the agent
        exchange (StockExchange): the exchange
        util_const (float): the constant in the utility function
        nrun (int): number of iterations
        report (boolean): True to return a list of cumulative_wealth over time
    '''    
    reward = 0
    state = (stock.get_price(), 0)    
    iter_count = 0
    cumulative_wealth = 0
    wealths = []
    
    for i in range(nrun):
        order = learner.learn(reward, state)        
        # when the exchange execute, it makes an impact on stock price
        transaction_cost = exchange.execute(order)        
        new_price, pnl = exchange.simulate_stock_price()
        
        delta_wealth = pnl - transaction_cost
        iter_count += 1
        cumulative_wealth += delta_wealth        
        
        reward = delta_wealth - util_const / 2 * (delta_wealth - cumulative_wealth / iter_count)**2
        state = (new_price, exchange.num_shares_owned)
        
        if report:
            wealths.append(cumulative_wealth)
        if i % 100000 == 0:
            print('finished 100k run')
    
    if report:
        return wealths
    else:
        return None

def main():    
    stock = OUStock(10.52, kappa=0.3, mu=16.0, sigma=0.25/252)
    lot = 100
    actions = list(range(-10*lot, 11*lot, lot))
    learner = QMatrix(actions)
    exchange = StockExchange(stock, lot=lot)        
    
    # for initial training and burn in
    run_stock_agent(stock, learner, exchange, nrun=int(1e7))    
    # for graphing pnl after training, run again the above 100k times
    wealths = run_stock_agent(stock, learner, exchange)
    
    plt.figure()
    plt.plot(range(len(wealths)), wealths)
    plt.title('Simple agent performance after initial training of 10 million steps')
    plt.xlabel('iterations')
    plt.ylabel('cumulative wealth')
    plt.savefig('first test.png')
    plt.show()
    
if __name__ == '__main__':
    main()
    print('all good')