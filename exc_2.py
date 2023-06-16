import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from exc_1 import datas
from random import *
import seaborn as sns
import matplotlib.dates as mdates

# ----------------------------------- Simulated VaR ----------------------------------- #


expected_return = [datas["AAPL_returns"].mean(), datas["MSFT_returns"].mean()]
volatility = [datas["AAPL_returns"].std(), datas["MSFT_returns"].std()]
correlation = 0.3


# The weight should be inversely proportional to the asset's volatility.
weights = [1 / volatility[0], 1 / volatility[1]]
weights = weights / np.sum(weights)
weights = np.array(weights)


def simulated_returns(expected_return=expected_return,
                      volatility=volatility,
                      correlation=correlation,
                      numOfSim=1000):

    dZA = np.random.normal(0, 1, size=numOfSim)
    dZB = np.random.normal(0, 1, size=numOfSim)
    dWA = dZA
    dWB = correlation * dZA + np.sqrt(1 - correlation ** 2) * dZB
    dln_S_A = (expected_return[0]-volatility[0]**2/2) + volatility[0]*dWA
    dln_S_B = (expected_return[1]-volatility[1]**2/2) + volatility[1]*dWB
    return np.array([dln_S_A, dln_S_B])


def calc_portfolio_var(weights=weights, alpha=0.05, corr=correlation):
    simulated_returns_ = simulated_returns(correlation=corr)
    portfolio_returns = np.matmul(weights, simulated_returns_)
    var = np.quantile(portfolio_returns, alpha)
    return var


sns.set(style='darkgrid')  # Set the seaborn style


def plot_simulated_portfolio_returns():
    for _ in range(20):
        simulated_returns_ = simulated_returns()
        portfolio_returns = np.matmul(weights, simulated_returns_)
        simulated_path = portfolio_returns.cumsum()
        plt.plot(simulated_path)

    plt.xlabel('Time')
    plt.ylabel('Portfolio Returns')
    plt.title('Simulated Portfolio Returns')
    plt.show()


def plot_vars_corr(corr=correlation):
    vars = []
    # x-values from -1 to 1 with step 0.01
    x_values = [i / 100 - 1 for i in range(0, 201)]

    for i in x_values:
        vars.append(calc_portfolio_var(corr=i))

    plt.plot(x_values, vars)
    plt.xlabel('Correlation')
    plt.ylabel('Portfolio VaR')
    plt.title('Portfolio VaR vs Correlation')
    plt.show()


def calc_portfolio_var_MC():
    s = 0
    for _ in range(10000):
        s += calc_portfolio_var()
    s /= 10000
    print(f"MC simulation Var: {s}")
    return s


plot_simulated_portfolio_returns()
plot_vars_corr()
calc_portfolio_var_MC()
