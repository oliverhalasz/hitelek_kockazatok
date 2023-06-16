import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import sys
import seaborn as sns

TICKER = "VOO"

# Step 1: Download historical price data
data = pd.DataFrame()


def get_data():
    global data
    stock = yf.Ticker(TICKER)
    close = stock.history(start='2010-01-01', end="2022-12-31")["Close"]
    data[f"{TICKER}"] = close


def calculate_daily_returns():
    global data
    data[f"{TICKER}_returns"] = data[f"{TICKER}"].pct_change()


get_data()
calculate_daily_returns()
data = data.dropna()

# Step 2: Calculate and plot the EWMA variance
def calculate_ewma_variance(df_etf_returns=data[[f"{TICKER}_returns"]],
                            decay_factor=0.94,
                            window=100):
    
    ewma = df_etf_returns.ewm(alpha=1-decay_factor,
                              min_periods=window,
                              adjust=False).var()
    return ewma




def plot_ewma_vars_sea():
    ewma_var_094 = calculate_ewma_variance(decay_factor=0.94)
    ewma_var_097 = calculate_ewma_variance(decay_factor=0.97)

    sns.set(style='darkgrid')  # Set the seaborn style

    plt.figure(figsize=(15, 10))
    plt.plot(ewma_var_094, label='EWMA Variance (Decay Factor = 0.94)')
    plt.plot(ewma_var_097, label='EWMA Variance (Decay Factor = 0.97)')
    plt.xlabel('Date')
    plt.ylabel('Variance')
    plt.title('EWMA Variance of ETF Returns')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_ewma_vars_sea()
    # Step 3: Reflections
    # print("Reflections on findings:")
    # print("The EWMA variance with a higher decay factor (0.97) has a smoother curve and reacts more slowly to changes compared to the EWMA variance with a lower decay factor (0.94).")
    # print("A higher decay factor places more weight on recent returns, resulting in a slower adjustment to new information.")
    # print("On the other hand, a lower decay factor gives more weight to older returns, making the variance more responsive to recent changes.")
    # print("The choice of decay factor depends on the desired level of sensitivity to recent returns and the time horizon of interest.")
