import warnings

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

REGRESSION_DATA_PATH = "./data/regression_data.csv"


def load_regression_data(historical_data, clusters_data, future_days, update):
    """Regression Data"""

    # import regression data
    if update == False:
        try:
            regression_data = pd.read_csv(REGRESSION_DATA_PATH)
        except FileNotFoundError:
            print("Regression data file not found.")
        else:
            return regression_data

    regression_data = (
        historical_data.merge(  # historical_data[historical_data['symbol'] == 'BTC']
            clusters_data, on="symbol", how="inner"
        )
    )

    # obtain future return and risk
    regression_data["future_return"] = (
        (1 + regression_data["return"])
        .rolling(window=future_days)
        .apply(np.prod, raw=True)
        - 1
    ).shift(1)
    regression_data["future_risk"] = (
        regression_data["return"].rolling(window=future_days).agg(np.var)
    )
    regression_data.loc[
        regression_data.groupby("symbol").cumcount() < future_days,
        ["future_return", "future_risk"],
    ] = np.nan
    regression_data.dropna(inplace=True)  # subset='future_return'
    regression_data.reset_index(drop=True, inplace=True)

    # normalize data
    regression_max_data = (
        regression_data[
            [
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "avg",
                "return",
                "volume",
                "marketcap",
                "future_return",
                "future_risk",
            ]
        ]
        .groupby("symbol")
        .max()
    )
    regression_max_data = regression_data[["symbol"]].merge(
        regression_max_data, on="symbol", how="left"
    )
    regression_normalized_data = (
        regression_data[
            [
                "future_return",
                "future_risk",
                "open",
                "high",
                "low",
                "close",
                "avg",
                "return",
                "volume",
                "marketcap",
            ]
        ]
        / regression_max_data[
            [
                "future_return",
                "future_risk",
                "open",
                "high",
                "low",
                "close",
                "avg",
                "return",
                "volume",
                "marketcap",
            ]
        ]
    )
    regression_data = regression_data[["symbol", "cluster"]].join(
        regression_normalized_data
    )
    regression_data["cluster"] = regression_data["cluster"].astype("str")

    regression_data = pd.get_dummies(regression_data)

    # export regression data
    regression_data.to_csv(REGRESSION_DATA_PATH, index=False)
    return regression_data


# def load_prediction_data(regression_data, model, update):
#     x = regression_data.loc[:, 'open':]
#     y = regression_data[['future_return', 'future_risk']]
#     train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2, random_state = 4)

#     match model:
#         case "linear_regression":
#             regr = linear_model.LinearRegression()
#         case "random_forest_regressor":
#             regr = RandomForestRegressor(n_estimators=100)
#         case _:
#             raise ValueError("Wrong model entered.")

#     regr.fit(train_x, train_y)

#     train_y_ = regr.predict(train_x)
#     print(f'Train R2: {(100 * r2_score(train_y, train_y_)):.2f}%')

#     test_y_ = regr.predict(test_x)
#     print(f'Test R2: {(100 * r2_score(test_y, test_y_)):.2f}%')
