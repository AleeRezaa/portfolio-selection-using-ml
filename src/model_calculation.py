import warnings

import numpy as np
import pandas as pd
from sklearn import cluster, covariance
from itertools import combinations
import warnings

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage, sample_cov
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import HRPOpt
from pypfopt.efficient_frontier import EfficientCVaR

import src.data_preparation as dp

warnings.filterwarnings("ignore")

CLUSTERS_DATA_PATH = "./data/clusters_data.csv"
REGRESSION_DATA_PATH = "./data/regression_data.csv"


# https://www.relataly.com/crypto-market-cluster-analysis-using-affinity-propagation-python/8114/
# Affinity Propagation انتشار وابستگی
def load_clusters_data(return_data, update=False):
    """Clusters Data"""

    # import clusters data
    if update == False:
        try:
            clusters_data = pd.read_csv(CLUSTERS_DATA_PATH)
        except FileNotFoundError:
            print("Clusters data file not found.")
        else:
            return clusters_data

    # create clustering data
    symbols = return_data.columns
    clustering_data = np.array(return_data / return_data.std())

    # create clustering modol
    edge_model = covariance.GraphicalLassoCV()
    edge_model.fit(clustering_data)
    cluster_centers_indices, labels = cluster.affinity_propagation(
        edge_model.covariance_, random_state=1
    )
    n = labels.max() + 1

    # create clusters
    clusters = []
    for i in range(n):
        sub_cluster = list(symbols[labels == i])
        clusters.append(sub_cluster)

    # create clusters data
    clusters_data = pd.DataFrame(columns=["symbol", "cluster"])
    for c in range(len(clusters)):
        for symbol in clusters[c]:
            clusters_data.loc[len(clusters_data)] = [
                symbol,
                str(c).zfill(len(str(n - 1))),
            ]

    # export clusters data
    clusters_data.to_csv(CLUSTERS_DATA_PATH, index=False)

    print("Clusters data file saved.")
    return clusters_data


def load_regression_data(normalized_data, model, update=False):
    # import regression data
    if update == False:
        try:
            regression_data = pd.read_csv(REGRESSION_DATA_PATH)
        except FileNotFoundError:
            print("Regression data file not found.")
        else:
            return regression_data

    regression_data = normalized_data.groupby("symbol").first().reset_index()
    regression_data.drop(columns=["future_return", "future_risk"], inplace=True)

    normalized_data.dropna(inplace=True)

    x = normalized_data.loc[:, "open_n" : normalized_data.columns[-3]]
    y = normalized_data[["future_return", "future_risk"]]
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

    x_pred = regression_data.loc[:, "open_n":]
    y_pred = regr.predict(x_pred)
    regression_data = regression_data.join(
        pd.DataFrame(y_pred, columns=["future_return", "future_risk"])
    )

    # export regression data
    regression_data.to_csv(REGRESSION_DATA_PATH, index=False)
    return regression_data


def load_dominated_data(regression_data):
    """Dominated Data"""

    dominated_data = regression_data[
        ["symbol", "cluster", "future_return", "future_risk"]
    ].copy()
    dominated_symbols = list(dominated_data["symbol"])

    for cluster, cluster_data in dominated_data.groupby("cluster"):
        cluster_combinations = combinations(cluster_data["symbol"], 2)
        for comb in list(cluster_combinations):
            symbol1 = comb[0]
            symbol2 = comb[1]
            return1 = cluster_data.loc[
                cluster_data["symbol"] == symbol1, "future_return"
            ].iloc[0]
            return2 = cluster_data.loc[
                cluster_data["symbol"] == symbol2, "future_return"
            ].iloc[0]
            risk1 = cluster_data.loc[
                cluster_data["symbol"] == symbol1, "future_risk"
            ].iloc[0]
            risk2 = cluster_data.loc[
                cluster_data["symbol"] == symbol2, "future_risk"
            ].iloc[0]
            if return1 < return2 and risk1 > risk2 and symbol1 in dominated_symbols:
                dominated_symbols.remove(symbol1)
            elif return1 > return2 and risk1 < risk2 and symbol2 in dominated_symbols:
                dominated_symbols.remove(symbol2)

    dominated_data = dominated_data[dominated_data["symbol"].isin(dominated_symbols)]
    dominated_data.sort_values(
        by=["cluster", "future_return"], ascending=[True, False], inplace=True
    )
    dominated_data.reset_index(drop=True, inplace=True)
    return dominated_data


def load_ew_portfolio(df):
    portfolio_df = df[["symbol"]].copy()
    portfolio_df["weight"] = 1 / portfolio_df.shape[0]
    return portfolio_df


def load_portfolio_with(model, close_data):
    match model:
        case "mv":
            m = mean_historical_return(close_data)
            s = sample_cov(
                close_data
            )  # s = CovarianceShrinkage(close_df).ledoit_wolf()
            md = EfficientFrontier(m, s)
            weights = md.max_sharpe()
        case "hrp":
            returns = close_data.pct_change().dropna()
            md = HRPOpt(returns)
            weights = md.optimize()
        case "mcvar":
            m = mean_historical_return(close_data)
            s = close_data.cov()
            md = EfficientCVaR(m, s)
            weights = md.min_cvar()

        case _:
            raise ValueError("Wrong model entered.")

    cleaned_weights = dict(md.clean_weights())
    md.portfolio_performance(verbose=True)
    portfolio_data = dp.get_portfolio_from_dict(dict(cleaned_weights))
    return portfolio_data


def get_portfolio_performance(portfolio_data, performance_data):
    performance_data = performance_data.sort_values("date")
    performance_data = performance_data[
        performance_data["symbol"].isin(portfolio_data["symbol"].unique())
    ]

    # Calculate the daily returns of each symbol
    returns_df = performance_data.pivot(
        index="date", columns="symbol", values="close"
    ).pct_change()

    # Calculate the weighted average daily return of the portfolio
    weights = portfolio_data.set_index("symbol")["weight"]
    portfolio_returns = (returns_df * weights).sum(axis=1)

    # Calculate the daily excess return of the portfolio
    risk_free_rate = 0
    excess_returns = portfolio_returns - risk_free_rate

    # Calculate the covariance matrix of the daily returns of the symbols in the portfolio
    covariance_matrix = returns_df.cov()

    # Calculate the standard deviation of the daily excess returns of the portfolio using the covariance matrix
    excess_returns_std = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

    # Calculate the Sharpe ratio of the portfolio
    sharpe_ratio = excess_returns.mean() / excess_returns_std
    return sharpe_ratio
