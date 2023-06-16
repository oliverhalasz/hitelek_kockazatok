def calculate_historical_var(portfolio_return, alpha):
    var = portfolio_return.squeeze().quantile(1 - alpha)
    return var


def simulated_returns(expected_return,
                      volatility,
                      correlation,
                      numOfSim=1000):

    try:
        global weights
    except NameError:
        weights = np.array([0.5, 0.5])

    dZA = np.random.normal(0, 1, size=numOfSim)
    dZB = np.random.normal(0, 1, size=numOfSim)

    dWA = dZA
    dWB = correlation * dZA + np.sqrt(1 - correlation ** 2) * dZB

    dln_S_A = (expected_return[0]-volatility[0]**2/2) + volatility[0]*dWA
    dln_S_B = (expected_return[1]-volatility[1]**2/2) + volatility[1]*dWB

    return np.matmul(weights, np.array([dln_S_A, dln_S_B]))


def calculate_ewma_variance(df_etf_returns,
                            decay_factor=0.94,
                            window=100):

    ewma = df_etf_returns.ewm(alpha=1-decay_factor,
                              min_periods=window,
                              adjust=False).var()
    return ewma

