# Portfolio VaR Analysis

This is a Python code snippet for portfolio analysis. The code uses various libraries such as `pandas`, `numpy`, `matplotlib`, and `yfinance`. It retrieves historical stock data, calculates daily returns, calculates portfolio returns, and calculates the historical Value at Risk (VaR) for the portfolio.

## Code Explanation


### Importing Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from random import *
```

In this section, the necessary libraries are imported for data manipulation, mathematical operations, visualization, and retrieving stock data using the Yahoo Finance API.

### Defining Tickers

```python
TICKERS = {
    "AAPL": 0.5,
    "MSFT": 0.5
}
```
The TICKERS dictionary contains the tickers (stock symbols) and their corresponding weights for the portfolio. In this example, the portfolio is equally weighted between Apple (AAPL) and Microsoft (MSFT).

### Retrieving Historical Data

```python
datas = pd.DataFrame()

def get_data(tickers=TICKERS):
    global datas
    for ticker in tickers.keys():
        stock = yf.Ticker(ticker)
        data = stock.history(start='2010-01-01', end="2022-12-31")["Close"]
        datas[f"{ticker}"] = data

get_data()
```

The get_data() function retrieves historical stock data for the specified tickers using the Yahoo Finance API. It stores the closing prices of each stock in a pandas DataFrame called datas.

### Calculating Daily Returns
```python
def calculate_daily_returns():
    global datas
    for ticker in TICKERS:
        datas[f"{ticker}_returns"] = datas[f"{ticker}"].pct_change()

calculate_daily_returns()
```

The calculate_daily_returns() function calculates the daily returns for each stock in the portfolio. It uses the pct_change() function to compute the percentage change in the closing prices and stores the results in new columns in the datas DataFrame.

### Calculating Portfolio Returns

```python
def calc_portfolio_return():
    global datas
    datas["portfolio_return"] = 0
    for ticker, weight in TICKERS.items():
        datas["portfolio_return"] += weight * datas[f"{ticker}_returns"]

calc_portfolio_return()
```

The calculate_daily_returns() function calculates the daily returns for each stock in the portfolio. It uses the pct_change() function to compute the percentage change in the closing prices and stores the results in new columns in the datas DataFrame.

### Calculating Portfolio Returns
    
```python
def calc_portfolio_return():
    global datas
    datas["portfolio_return"] = 0
    for ticker, weight in TICKERS.items():
        datas["portfolio_return"] += weight * datas[f"{ticker}_returns"]

calc_portfolio_return()
```

The calc_portfolio_return() function calculates the portfolio returns by multiplying the daily returns of each stock by its corresponding weight and summing them. The results are stored in a new column called "portfolio_return" in the datas DataFrame.

### Calculating Historical VaR
    
```python
def calc_portfolio_var(portfolio_return=datas[["portfolio_return"]], alpha=0.95):
    var = portfolio_return.squeeze().quantile(1 - alpha)
    return var

print(calc_portfolio_var())
```
The calc_portfolio_var() function calculates the historical Value at Risk (VaR) for the portfolio. It uses the quantile() function to determine the portfolio return value below which the specified percentile (1 - alpha) lies. The result is the VaR at the specified confidence level. Finally, the VaR value is printed along with the datas DataFrame containing the portfolio data.

## Conclusion
This code snippet demonstrates a simple implementation of portfolio analysis using historical stock data. It retrieves data, calculates daily returns, computes portfolio returns, and determines the historical VaR for the portfolio. Feel free to modify the code according to your requirements and explore additional analysis techniques.

# Simulated Portfolio VaR Analysis

This is a Python code snippet for simulated Value at Risk (VaR) analysis. The code uses `numpy`, `pandas`, and `matplotlib` libraries. It calculates simulated returns, defines portfolio weights, and calculates the VaR of the portfolio.

## Code Explanation

The provided code performs the following steps:

### Importing Libraries and Data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from exc_1 import datas
from random import *
```
### Exploring Data
    
```python
print(datas.describe())
```

### Defining Expected Return, Volatility, and Correlation
    
```python
expected_return = [datas["AAPL_returns"].mean(), datas["MSFT_returns"].mean()]
volatility = [datas["AAPL_returns"].std(), datas["MSFT_returns"].std()]
correlation = 0.3
```
The expected return and volatility are calculated based on the mean and standard deviation of daily returns for Apple (AAPL) and Microsoft (MSFT) stocks. The correlation represents the relationship between the two stocks.

### Defining Portfolio Weights

```python
weights = [1/volatility[0], 1/volatility[1]]
weights = weights/np.sum(weights)

print(weights)
```
The portfolio weights are defined inversely proportional to the volatility of each stock. The weights are normalized to ensure their sum is equal to 1. The resulting weights are printed.


### Simulating Returns
    
```python
def simulated_returns(expected_return=expected_return,
                      volatility=volatility,
                      correlation=correlation,
                      numOfSim=1000):

    dZA = np.random.normal(0, 1, size=numOfSim)
    dZB = np.random.normal(0, 1, size=numOfSim)
    dWA = dZA
    dWB = correlation*dZA + np.sqrt(1-correlation**2)*dZB
    dln_S_A = (expected_return[0]-volatility[0]**2/2) + volatility[0]*dWA
    dln_S_B = (expected_return[1]-volatility[1]**2/2) + volatility[1]*dWB

    plt.plot(dln_S_A)
    plt.show()
    return np.array([dln_S_A, dln_S_B])
```
The simulated_returns() function simulates the logarithmic returns of the two stocks based on the provided expected return, volatility, correlation, and number of simulations. It generates random values (dZA and dZB) from a normal distribution, and then calculates the logarithmic returns (dln_S_A and dln_S_B) using the provided formulas. The resulting simulated returns are plotted for the first stock (dln_S_A).

### Calculating Portfolio VaR
    
```python
def calc_portfolio_var(weights=weights, alpha=0.05):
    simulated_returns_ = simulated_returns()
    portfolio_returns = np.matmul(weights, simulated_returns_)
    var = np.quantile(portfolio_returns, alpha)
    return var

```
This code snippet demonstrates how to calculate the Value at Risk (VaR) for the portfolio using simulated returns. The calc_portfolio_var() function takes the portfolio weights and the desired confidence level (alpha) as inputs. It calls the simulated_returns() function to obtain an array of simulated returns. The portfolio returns are calculated by multiplying the weights with the simulated returns using np.matmul(). Finally, the VaR is computed using the np.quantile() function, specifying the desired confidence level (alpha).

## Conclusion
This code snippet demonstrates how to perform simulated Value at Risk (VaR) analysis for a portfolio. It uses simulated returns based on expected returns, volatilities, and correlation between assets. The portfolio weights are defined based on asset volatility, and the VaR is calculated using the simulated returns and portfolio weights.

# Analysis of EWMA Variance of ETF Returns

## Importing Required Libraries

We begin by importing the necessary libraries for our analysis: pandas, matplotlib.pyplot, yfinance, sys, and seaborn.

```python
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import sys
import seaborn as sns
```

## Importing Data

```python
TICKER = "VOO"
data = pd.DataFrame()
def get_data():
    global data
    stock = yf.Ticker(TICKER)
    close = stock.history(start='2010-01-01', end="2022-12-31")["Close"]
    data[f"{TICKER}"] = close
```
We define a function get_data() to download the historical price data using the yfinance library. It retrieves the closing prices for the specified ETF from January 1, 2010, to December 31, 2022, and stores them in the data DataFrame under the column with the ticker symbol.

Next, we define a function calculate_daily_returns() to calculate the daily returns of the ETF. It adds a new column to the data DataFrame called {TICKER}_returns, which represents the percentage change in the closing prices.

```python
def calculate_daily_returns():
    global data
    data[f"{TICKER}_returns"] = data[f"{TICKER}"].pct_change()
```

We call the get_data() and calculate_daily_returns() functions to download the data and calculate the daily returns.

Finally, we drop any rows with missing values (NaN) from the data DataFrame.

```python
get_data()
calculate_daily_returns()
data = data.dropna()
```

## Calculate and Plot the EWMA Variance
We define a function calculate_ewma_variance() to calculate the exponentially weighted moving average (EWMA) variance of the ETF returns. The function takes parameters df_etf_returns (defaulting to the data DataFrame), decay_factor (defaulting to 0.94), and window (defaulting to 100).
```python
def calculate_ewma_variance(df_etf_returns=data[[f"{TICKER}_returns"]],
                            decay_factor=0.94,
                            window=100):
    ewma = df_etf_returns.ewm(alpha=1-decay_factor,
                              min_periods=window,
                              adjust=False).var()
    return ewma
```
Next, we define a function plot_ewma_vars_sea() to plot the EWMA variances using seaborn. It calls the calculate_ewma_variance() function twice with different decay factors to obtain two sets of variances. It then uses seaborn and matplotlib to create a line plot of the EWMA variances.

```python
def plot_ewma_vars_sea():
    ewma_var_094 = calculate_ewma_variance(decay_factor=0.94)
    ewma_var_097 = calculate_ewma_variance(decay_factor=0.97)

    sns.set(style='darkgrid')  # Set the seaborn style

    plt.figure(figsize=(15, 10))
```

