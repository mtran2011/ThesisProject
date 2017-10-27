import matplotlib.pyplot as plt
from stock import OUStock
from exchange import StockExchange
from qlearner import *
from environment import StockTradingEnvironment
import distance_func
# import model_builder

def graph_performance(wealths_list, agent_names):
    plt.figure()
    for wealths, agent in zip(wealths_list, agent_names):
        plt.plot(range(len(wealths)), wealths, label=agent)
        
    plt.title('Agent performance in {0} steps after training'.format(len(wealths_list[0])))
    plt.legend(loc='best')
    plt.xlabel('iterations')
    plt.ylabel('cumulative wealth')
    plt.savefig('../figs/newfig.png')    

def make_exchange():
    stock = OUStock(price=110, kappa=0.1, mu=125, sigma=0.2, tick=0.1, band=1000)
    lot = 100
    actions = list(range(-3*lot, 4*lot, lot))    
    exchange = StockExchange(stock, lot=lot, impact=0, max_holding=1000)
    return actions, exchange

def run_qmatrix_stock_trading():
    actions, exchange = make_exchange()
    learner = QMatrix(actions, epsilon=0.1, learning_rate=0.5, discount_factor=0.999)
    environment = StockTradingEnvironment(learner, exchange)
    
    # for initial training and burn in
    environment.run(1e-3, int(1e4))
    # for graphing pnl after training, run again the above 5k times
    wealths = environment.run(1e-3, 5000, report=True)
    graph_performance([wealths], ['qmatrix'])

def run_heuristic_qmatrix_stock_trading():
    stock = OUStock(price=110, kappa=0.1, mu=150, sigma=0.2, tick=0.1, band=1000)
    lot = 100
    actions = list(range(-3*lot, 4*lot, lot))
    exchange = StockExchange(stock, lot=lot, impact=0, max_holding=1000)
    ntrain = int(1e6)
    ntest = 5000
    util = 1e-3

    # for simple QMatrix    
    learner = QMatrix(actions, epsilon=0.1, learning_rate=0.5, discount_factor=0.999)
    environment = StockTradingEnvironment(stock, learner, exchange)
    environment.run(util, ntrain)
    wealths_qmatrix = environment.run(util, ntest, report=True)
    
    # for QMatrixHeuristic
    dist_func = lambda x1, x2: distance_func.p_norm(x1, x2, p=2)
    learner = QMatrixHeuristic(actions, dist_func, epsilon=0.1, learning_rate=0.5, discount_factor=0.999)
    environment = StockTradingEnvironment(stock, learner, exchange)
    environment.run(util, ntrain)
    wealths_qheuristic = environment.run(util, ntest, report=True)
        
    graph_performance([wealths_qmatrix, wealths_qheuristic], ['simple Q matrix', 'heuristic Q matrix'])

def run_dqn_stock_trading():
    stock = OUStock(price=110, kappa=0.1, mu=150, sigma=0.2, tick=0.1, band=1000)
    lot = 100
    actions = list(range(-3*lot, 4*lot, lot))    
    exchange = StockExchange(stock, lot=lot, impact=0, max_holding=1000)
    model = model_builder.build_simple_ff(len(actions), 2, len(actions))
    learner = DQNLearner(actions, model, epsilon=0.1, discount_factor=0.999)
    environment = StockTradingEnvironment(stock, learner, exchange)
    
    # for initial training and burn in
    environment.run(1e-3, int(1e4))
    # for graphing pnl after training, run again the above 5k times
    wealths = environment.run(1e-3, 5000, report=True)
    graph_performance([wealths], ['simple_dqn_ff'])
    
if __name__ == '__main__':
    run_heuristic_qmatrix_stock_trading()