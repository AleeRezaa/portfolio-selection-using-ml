import warnings

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

REGRESSION_DATA_PATH = "./data/regression_data.csv"
PREDICTION_DATA_PATH = "./data/prediction_data.csv"


def load_regression_data(
    selected_data,
    clusters_data,
    future_days=30,
    update=False,
):
    """Regression Data"""

    # import regression data
    if update == False:
        try:
            regression_data = pd.read_csv(REGRESSION_DATA_PATH)
        except FileNotFoundError:
            print("Regression data file not found.")
        else:
            return regression_data

    # add clusters
    regression_data = selected_data.merge(clusters_data, on="symbol", how="inner")

    # add date
    regression_data["date"] = pd.to_datetime(regression_data["date"])
    regression_data["year"] = regression_data["date"].dt.year
    regression_data["month_sin"] = np.sin(
        2 * np.pi * regression_data["date"].dt.month / 12
    )
    regression_data["month_cos"] = np.cos(
        2 * np.pi * regression_data["date"].dt.month / 12
    )
    regression_data["day_sin"] = np.sin(
        2 * np.pi * regression_data["date"].dt.month / 31
    )
    regression_data["day_cos"] = np.cos(
        2 * np.pi * regression_data["date"].dt.month / 31
    )

    # normalize input
    input_max_data = (
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
                "year",
                "month_sin",
                "month_cos",
                "day_sin",
                "day_cos",
            ]
        ]
        .groupby("symbol")
        .max()
    )
    input_max_data = regression_data[["symbol"]].merge(
        input_max_data, on="symbol", how="left"
    )
    input_normalized_data = (
        regression_data[
            [
                "open",
                "high",
                "low",
                "close",
                "avg",
                "return",
                "volume",
                "marketcap",
                "year",
                "month_sin",
                "month_cos",
                "day_sin",
                "day_cos",
            ]
        ]
        / input_max_data[
            [
                "open",
                "high",
                "low",
                "close",
                "avg",
                "return",
                "volume",
                "marketcap",
                "year",
                "month_sin",
                "month_cos",
                "day_sin",
                "day_cos",
            ]
        ]
    )
    input_normalized_data.columns = input_normalized_data.columns + "_n"
    regression_data = regression_data.join(input_normalized_data)

    # get dummies of symbols and clusters
    regression_data["cluster"] = regression_data["cluster"].astype("str")
    regression_data = regression_data[["symbol"]].join(pd.get_dummies(regression_data))

    # obtain future return and risk
    regression_data["future_return"] = (
        (1 + regression_data["return_n"])
        .rolling(window=future_days)
        .apply(np.prod, raw=True)
        - 1
    ).shift(1)
    regression_data["future_risk"] = (
        regression_data["return_n"].rolling(window=future_days).agg(np.var)
    )
    regression_data.loc[
        regression_data.groupby("symbol").cumcount() < future_days,
        ["future_return", "future_risk"],
    ] = np.nan
    # regression_data.dropna(inplace=True)  # subset='future_return'
    regression_data.reset_index(drop=True, inplace=True)

    # export regression data
    regression_data.to_csv(REGRESSION_DATA_PATH, index=False)
    return regression_data


def load_prediction_data(regression_data, model, update=False):
    # import regression data
    if update == False:
        try:
            prediction_data = pd.read_csv(PREDICTION_DATA_PATH)
        except FileNotFoundError:
            print("Prediction data file not found.")
        else:
            return prediction_data

    prediction_data = regression_data.groupby("symbol").first().reset_index()
    prediction_data.drop(columns=["future_return", "future_risk"], inplace=True)

    regression_data.dropna(inplace=True)

    x = regression_data.loc[:, "open_n" : regression_data.columns[-3]]
    y = regression_data[["future_return", "future_risk"]]
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.2, random_state=4
    )

    match model:
        case "linear_regression":
            regr = linear_model.LinearRegression()
        case "random_forest_regressor":
            regr = RandomForestRegressor(n_estimators=100)
        case _:
            raise ValueError("Wrong model entered.")

    regr.fit(train_x, train_y)

    train_y_ = regr.predict(train_x)
    print(f"Train R2: {(100 * r2_score(train_y, train_y_)):.2f}%")

    test_y_ = regr.predict(test_x)
    print(f"Test R2: {(100 * r2_score(test_y, test_y_)):.2f}%")

    x_pred = prediction_data.loc[:, "open_n":]
    y_pred = regr.predict(x_pred)
    prediction_data = prediction_data.join(
        pd.DataFrame(y_pred, columns=["future_return", "future_risk"])
    )

    # export regression data
    prediction_data.to_csv(PREDICTION_DATA_PATH, index=False)
    return prediction_data
