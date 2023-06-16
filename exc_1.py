import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from random import *

TICKERS = {
    "AAPL": 0.5,
    "MSFT": 0.5
}

datas = pd.DataFrame()


def get_data(tickers=TICKERS):
    global datas
    for ticker in tickers.keys():
        stock = yf.Ticker(ticker)
        data = stock.history(start='2010-01-01', end="2022-12-31")["Close"]
        datas[f"{ticker}"] = data


def calculate_daily_returns():
    global datas
    for ticker in TICKERS:
        datas[f"{ticker}_returns"] = datas[f"{ticker}"].pct_change()


def calc_portfolio_return():
    global datas
    datas["portfolio_return"] = 0
    for ticker, weight in TICKERS.items():
        datas["portfolio_return"] += weight * datas[f"{ticker}_returns"]


get_data()
calculate_daily_returns()
calc_portfolio_return()
datas = datas.dropna()


def calc_portfolio_var(portfolio_return=datas[["portfolio_return"]], alpha=0.95):
    var = portfolio_return.squeeze().quantile(1 - alpha)
    return var


if __name__ == "__main__":
    print(datas)
    print(f"Portfolio VaR: {calc_portfolio_var()}")
