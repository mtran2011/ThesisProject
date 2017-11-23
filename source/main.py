import itertools
from math import log
import matplotlib.pyplot as plt
from stock import OUStock, OULogStock, GBMStock
from exchange import StockExchange, StockOptionExchange
from qlearner import TabularQMatrix, KernelSmoothingQMatrix
from environment import StockTradingEnvironment, OptionHedgingEnvironment
from function_estimator import PairwiseLinearEstimator
from option import EuropeanStockOption
import kernel_function

def graph_performance(wealths_list, agent_names, ntrain):
    linestyles = itertools.cycle(['-', '--', '-.', ':'])
    colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    
    plt.figure()
    for wealths, agent_name in zip(wealths_list, agent_names):
        plt.plot(range(len(wealths)), wealths, label=agent_name, 
                 linestyle=next(linestyles),
                 color=next(colors))
    
    ntest = len(wealths_list[0])
    plt.title('Performance with ntrain = {0:,} and ntest = {1:,}'.format(ntrain, ntest))
    plt.legend(loc='best')
    plt.xlabel('iterations of testing runs')
    plt.ylabel('performance measure')
    plt.savefig('../figs/newfig.png')

def make_stock_exchange():
    stock = OULogStock(price=10, kappa=0.1, mu=log(15), sigma=0.2, tick=0.1, band=1000)
    lot = 10
    actions = list(range(-3*lot, 4*lot, lot))
    exchange = StockExchange(stock, lot=lot, impact=0, max_holding=15*lot)
    return actions, exchange

def make_option_exchange():
    stock = GBMStock(price=100, mu=0.001, sigma=0.01, tick=0.01, band=1000)
    option = EuropeanStockOption(stock, strike=105, expiry=252, is_call=True)
    lot = 10
    actions = list(range(-5*lot, 6*lot, lot))
    exchange = StockOptionExchange(option, lot=lot, impact=0, max_holding=5*lot)
    return actions, exchange

def run_qmatrix_stock_trading():
    actions, exchange = make_stock_exchange()
    util, ntrain, ntest = 1e-3, int(1e5), 5000
    
    # for SemiGradQLearner
    '''
    qfunc_estimator = PairwiseLinearEstimator(num_state_features=2)
    qgrad_learner = SemiGradQLearner(actions, qfunc_estimator, epsilon=0.1, learning_rate=1e-5, discount_factor=0.999)
    environment = StockTradingEnvironment(qgrad_learner, exchange)
    environment.run(util, ntrain)    
    wealths_semigrad = environment.run(util, ntest, report=True)
    print(qfunc_estimator.get_params())
    '''

    epsilon, learning_rate, discount_factor = 0.1, 0.5, 0.999
    # for simple TabularQMatrix
    tabular_qlearner = TabularQMatrix(actions, epsilon, learning_rate, discount_factor)
    environment = StockTradingEnvironment(tabular_qlearner, exchange)
    environment.run(util, ntrain)
    wealths_tabular_qmatrix = environment.run(util, ntest, report=True)

    # for KernelSmoothingQMatrix using inverse L2 distance
    kernel = lambda x1, x2: kernel_function.inverse_norm_p(x1, x2, p=2)
    smoothing_qlearner = KernelSmoothingQMatrix(actions, kernel, epsilon, learning_rate, discount_factor)
    environment = StockTradingEnvironment(smoothing_qlearner, exchange)
    environment.run(util, ntrain)
    wealths_smoothing_qmatrix = environment.run(util, ntest, report=True)

    graph_performance([wealths_tabular_qmatrix, wealths_smoothing_qmatrix],
                      ['tabular Q matrix', 'smoothing Q matrix'], ntrain)

def run_qmatrix_option_hedging():
    actions, exchange = make_option_exchange()
    util, ntrain, ntest = 1e-3, int(2e4), 1000
    epsilon, learning_rate, discount_factor = 0.1, 0.5, 0.999
    # for KernelSmoothingQMatrix using inverse L2 distance
    kernel = lambda x1, x2: kernel_function.inverse_norm_p(x1, x2, p=2)
    smoothing_qlearner = KernelSmoothingQMatrix(actions, kernel, epsilon, learning_rate, discount_factor)
    environment = OptionHedgingEnvironment(smoothing_qlearner, exchange)
    environment.run(util, ntrain)
    wealths = environment.run(util, ntest, report=True)

    graph_performance([wealths], ['inverse L2 norm smoothing'], ntrain)

'''
def run_dqn_stock_trading():
    actions, exchange = make_stock_exchange()
    util, ntrain, ntest = 1e-3, int(1e6), 5000

    model = model_builder.build_simple_ff(len(actions), 2, len(actions))
    learner = DQNLearner(actions, model, epsilon=0.1, discount_factor=0.999)
    environment = StockTradingEnvironment(learner, exchange)

    environment.run(util, ntrain)    
    wealths = environment.run(util, ntest, report=True)
    graph_performance([wealths], ['simple_dqn_feed_forward'], ntrain)    
'''

if __name__ == '__main__':
    # run_qmatrix_stock_trading()
    run_qmatrix_option_hedging()