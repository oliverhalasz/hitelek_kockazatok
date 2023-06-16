from exc_3 import data, TICKER
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
import seaborn as sns

# Create lagged squared returns
# Maximum number of lags is 20 or (number of data points - 1)
data[f"{TICKER}_squared_returns"] = data[f"{TICKER}_returns"] ** 2


def create_lagged_squared_returns():
    global data
    lags = 20
    for i in range(1, lags + 1):
        data[f'Lagged Squared Returns {i}'] = data[f"{TICKER}_squared_returns"].shift(
            i)
    data = data.dropna()


create_lagged_squared_returns()


# Dependent variable and independent variables
X = data.drop(["VOO", "VOO_returns", "VOO_squared_returns"], axis=1)
y = data["VOO_squared_returns"]


# k-fold CV
model = LinearRegression()

tscv = TimeSeriesSplit(n_splits=20)

scores = -cross_val_score(model, X, y,
                          scoring='neg_mean_squared_error',
                          cv=tscv)


model.fit(X, y)

# Predict variance for the entire data
y_pred_all = model.predict(X)

# Plot the predicted variance vs. actual variance




sns.set(style='darkgrid')  # Set the seaborn style

def plot_predicted_vs_actual_variance2(data, TICKER, y_pred_all):
    plt.plot(data.index, data[f"{TICKER}_squared_returns"],
             label='Actual Variance')
    plt.plot(data.index, y_pred_all, label='Predicted Variance')
    plt.xlabel('Time')
    plt.ylabel('Variance')
    plt.title('Predicted vs Actual Variance')
    plt.legend()
    plt.show()


plot_predicted_vs_actual_variance2(data=data, TICKER=TICKER, y_pred_all=y_pred_all)