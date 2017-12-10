import itertools
from math import log
import matplotlib.pyplot as plt
from stock import OULogStock, GBMStock
from exchange import StockExchange, OptionHedgingExchange
from qlearner import TabularQMatrix, KernelSmoothingQMatrix
from sarsa import TabularSarsaMatrix, KernelSmoothingSarsaMatrix, RandomForestSarsaMatrix
from environment import StockTradingEnvironment, OptionHedgingEnvironment, GammaScalpingEnvironment
from option import Pair
from regressor import InverseNormWeighter

def graph_performance(wealths_list, agent_names, ntrain):
    linestyles = itertools.cycle(['-', ':', '--', '-.'])
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
    actions = tuple(range(-3*lot, 4*lot, lot))
    exchange = StockExchange(stock, lot=lot, impact=0, max_holding=15*lot)
    return actions, exchange

def run_qmatrix_stock_trading():
    actions, exchange = make_stock_exchange()
    util, ntrain, ntest = 1e-3, int(1e5), 5000
    
    # for SemiGradQLearner    
    # qfunc_estimator = PairwiseLinearEstimator(num_state_features=2)
    # qgrad_learner = SemiGradQLearner(actions, qfunc_estimator, epsilon=0.1, learning_rate=1e-5, discount_factor=0.999)
    # environment = StockTradingEnvironment(qgrad_learner, exchange)
    # environment.run(util, ntrain)    
    # wealths_semigrad = environment.run(util, ntest, report=True)
    # print(qfunc_estimator.get_params())
    

    epsilon, learning_rate, discount_factor = 0.1, 0.5, 0.999
    # for simple TabularQMatrix
    tabular_qmatrix = TabularQMatrix(actions, epsilon, learning_rate, discount_factor)
    environment = StockTradingEnvironment(tabular_qmatrix, exchange)
    environment.run(util, ntrain)
    wealths_tabular_qmatrix = environment.run(util, ntest, report=True)

    # for KernelSmoothingQMatrix using inverse L2 distance
    # kernel_func = lambda x1, x2: kernel.inverse_norm_p(x1, x2, p=2)
    # smoothing_qlearner = KernelSmoothingQMatrix(actions, kernel_func, epsilon, learning_rate, discount_factor)
    # environment = StockTradingEnvironment(smoothing_qlearner, exchange)
    # environment.run(util, ntrain)
    # wealths_smoothing_qmatrix = environment.run(util, ntest, report=True)

    tabular_sarsa = TabularSarsaMatrix(actions, epsilon, learning_rate, discount_factor)
    environment = StockTradingEnvironment(tabular_sarsa, exchange)
    environment.run(util, ntrain)
    wealths_tabular_sarsa = environment.run(util, ntest, report=True)

    # for kernel smoothing SARSA using inverse norm-1
    inverse_norm_weighter = InverseNormWeighter(p=1)
    inverse_norm_sarsa = KernelSmoothingSarsaMatrix(actions, inverse_norm_weighter, epsilon, learning_rate, discount_factor)
    environment = StockTradingEnvironment(inverse_norm_sarsa, exchange)
    environment.run(util, ntrain)
    wealths_weighting_sarsa = environment.run(util, ntest, report=True)

    graph_performance([wealths_tabular_qmatrix, wealths_tabular_sarsa, wealths_weighting_sarsa],
                      ['tabular Q matrix', 'tabular SARSA', 'inverse norm-1 weighting SARSA'], ntrain)

def make_option_exchange():
    stock = GBMStock(price=50, mu=0, sigma=0.03, tick=1, band=20)
    pair = Pair(stock, strike=50, expiry=53, iv=stock.sigma, is_call=True)
    lot = 1
    actions = tuple(range(-5*lot, 6*lot, lot))
    exchange = OptionHedgingExchange(pair, lot=lot, impact=0, max_holding=10*lot)
    return actions, exchange

def make_underpriced_option():
    stock = GBMStock(price=1000, mu=0, sigma=0.045, tick=0.01, band=int(1e6))
    pair = Pair(stock, strike=1000, expiry=252, iv=stock.sigma/3, is_call=True)
    lot = 1
    actions = tuple(range(-5*lot, 6*lot, lot))
    exchange = OptionHedgingExchange(pair, lot=lot, impact=0, max_holding=10*lot)
    return actions, exchange

def run_qmatrix_option_hedging():
    actions, exchange = make_option_exchange()
    util, ntrain, ntest = 1e-3, int(5e6), 10800
    epsilon, learning_rate, discount_factor = 0.1, 0.5, 0.9999
    
    # for tabular Q matrix    
    tabular_qmatrix = TabularQMatrix(actions, epsilon, learning_rate, discount_factor)
    environment = OptionHedgingEnvironment(tabular_qmatrix, exchange)
    environment.run(util, ntrain)
    rewards, average_rewards = environment.run(util, ntest, report=True)

    graph_performance([rewards, average_rewards], ['one-step reward for tabular Q-matrix', 'average reward for tabular Q-matrix'], ntrain)

def run_gamma_scalping():
    actions, exchange = make_underpriced_option()
    util, ntrain, ntest = 1e-3, int(5e3), 4*252
    epsilon, learning_rate, discount_factor = 0.1, 0.5, 0.9999

    # for tabular q matrix
    tabular_qmatrix = TabularQMatrix(actions, epsilon, learning_rate, discount_factor)
    environment = GammaScalpingEnvironment(tabular_qmatrix, exchange)
    environment.run(util, ntrain)
    wealths_tabular_qmatrix = environment.run(util, ntest, report=True)

    # for random forest SARSA
    rf_sarsa_learner = RandomForestSarsaMatrix(actions, epsilon, learning_rate, discount_factor)
    environment = GammaScalpingEnvironment(rf_sarsa_learner, exchange)
    environment.run(util, ntrain)
    wealths_rf_sarsa = environment.run(util, ntest, report=True)

    graph_performance(
        [wealths_tabular_qmatrix, wealths_rf_sarsa], 
        ['gamma scalping with tabular Q matrix', 'gamma scalping with random forest SARSA'], ntrain)

# def run_dqn_stock_trading():
#     actions, exchange = make_stock_exchange()
#     util, ntrain, ntest = 1e-3, int(1e6), 5000

#     model = model_builder.build_simple_ff(len(actions), 2, len(actions))
#     learner = DQNLearner(actions, model, epsilon=0.1, discount_factor=0.999)
#     environment = StockTradingEnvironment(learner, exchange)

#     environment.run(util, ntrain)    
#     wealths = environment.run(util, ntest, report=True)
#     graph_performance([wealths], ['simple_dqn_feed_forward'], ntrain)

if __name__ == '__main__':
    # run_qmatrix_stock_trading()
    run_qmatrix_option_hedging()
    # run_gamma_scalping()