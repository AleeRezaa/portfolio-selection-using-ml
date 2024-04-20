import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from pypfopt import HRPOpt
from pypfopt.efficient_frontier import EfficientCVaR, EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage, sample_cov
from sklearn import covariance, linear_model
from sklearn.cluster import affinity_propagation, KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import src.data_preparation as dp

warnings.filterwarnings("ignore")

CLUSTERS_DATA_PATH = "./data/clusters_data.csv"
REGRESSION_DATA_PATH = "./data/regression_data.csv"


def affinity_propagation_model(return_data: pd.DataFrame) -> pd.DataFrame:
    """
    Affinity Propagation انتشار وابستگی
    https://www.relataly.com/crypto-market-cluster-analysis-using-affinity-propagation-python/8114/
    """

    # create clustering data
    symbols = return_data.columns
    clustering_data = np.array(return_data / return_data.std())

    # create clustering modol
    edge_model = covariance.GraphicalLassoCV()
    edge_model.fit(clustering_data)
    cluster_centers_indices, labels = affinity_propagation(
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

    return clusters_data


def k_means_model(return_data: pd.DataFrame) -> pd.DataFrame:
    return_data_t = return_data.T
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(return_data_t)

    # Find optimal number of clusters using Silhouette Score
    silhouette_scores = []
    max_clusters = min(
        10, len(return_data_t.columns)
    )  # Limit to avoid excessive clusters
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(standardized_data)
        silhouette_scores.append(silhouette_score(standardized_data, kmeans.labels_))

    # Choose the number of clusters with the highest silhouette score
    optimal_clusters = (
        silhouette_scores.index(max(silhouette_scores)) + 2
    )  # Add 2 due to range start

    # Fit K-Means model with optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    kmeans.fit(standardized_data)

    # Add cluster labels to the original DataFrame
    return_data_t["cluster"] = kmeans.labels_
    clusters_data = pd.DataFrame(return_data_t["cluster"]).reset_index(names="symbol")
    return clusters_data


def load_clusters_data(return_data: pd.DataFrame, model: str) -> pd.DataFrame:
    """Clusters Data"""

    match model:
        case "affinity_propagation":
            clusters_data = affinity_propagation_model(return_data)
        case "k_means":
            clusters_data = k_means_model(return_data)
        case _:
            raise ValueError("Wrong model entered.")

    return clusters_data


def load_regression_data(
    normalized_data: pd.DataFrame, model: str, update: bool = False
) -> pd.DataFrame:
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


def calculate_risk_return(processed_data: pd.DataFrame, rf: float = 0) -> pd.DataFrame:
    risk_return_data = processed_data.groupby(by="symbol", as_index=False).apply(
        lambda x: pd.Series(
            {"return": (1 + x["return"]).prod() - 1, "risk": x["return"].var()}
        )
    )
    risk_return_data["sharpe_ratio"] = (risk_return_data["return"] - rf) / (
        risk_return_data["return"] ** 0.5
    )
    return risk_return_data


def select_symbols(df: pd.DataFrame, model: str, dominate: bool = True) -> list:
    selected_symbols = list(df["symbol"])
    if dominate:
        for cluster, cluster_data in df.groupby("cluster"):
            cluster_combinations = combinations(cluster_data["symbol"], 2)
            for comb in list(cluster_combinations):
                symbol1, symbol2 = comb[0], comb[1]
                return1 = cluster_data.loc[
                    cluster_data["symbol"] == symbol1, "return"
                ].iloc[0]
                return2 = cluster_data.loc[
                    cluster_data["symbol"] == symbol2, "return"
                ].iloc[0]
                risk1 = cluster_data.loc[
                    cluster_data["symbol"] == symbol1, "risk"
                ].iloc[0]
                risk2 = cluster_data.loc[
                    cluster_data["symbol"] == symbol2, "risk"
                ].iloc[0]
                if return1 < return2 and risk1 > risk2 and symbol1 in selected_symbols:
                    selected_symbols.remove(symbol1)
                elif (
                    return1 > return2 and risk1 < risk2 and symbol2 in selected_symbols
                ):
                    selected_symbols.remove(symbol2)
    df = df[df["symbol"].isin(selected_symbols)]

    match model:
        case "keep_all":
            return selected_symbols
        case "max_return":
            df = df.groupby(["cluster"]).apply(
                lambda group: group[group["return"] == group["return"].max()]
            )
        case "min_risk":
            df = df.groupby(["cluster"]).apply(
                lambda group: group[group["risk"] == group["risk"].min()]
            )
        case "max_sharpe":
            df = df.groupby(["cluster"]).apply(
                lambda group: group[
                    group["sharpe_ratio"] == group["sharpe_ratio"].max()
                ]
            )
        case _:
            raise ValueError("Wrong model entered.")

    selected_symbols = df["symbol"].unique()

    # TODO: fix issue if more symbols than clusters because of equal values
    if len(selected_symbols) != len(df["cluster"].unique()):
        raise ValueError("More symbols than clusters.")

    return selected_symbols


def sparse_portfolio(risk_return_data: pd.DataFrame):
    symbols = risk_return_data["symbol"]
    returns = risk_return_data["return"].to_numpy()
    risks = risk_return_data["risk"].to_numpy()

    def project_simplex(w):
        u = np.sort(w)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1.0)
        return np.maximum(w - theta, 0)

    def threshold(w, k):
        idx = np.argsort(w)[::-1][:k]
        w_new = np.zeros_like(w)
        w_new[idx] = w[idx]
        return w_new

    k = 10
    r = 0.001
    w = np.ones(len(symbols)) / len(symbols)
    alpha = 0.01
    tol = 1e-6
    max_iter = 10000
    iter = 0
    converged = False
    while not converged and iter < max_iter:
        w_old = w.copy()
        w = w - alpha * np.diag(risks) @ w
        w = project_simplex(w)
        w = threshold(w, k)
        if w.T @ returns < r:
            w = w_old.copy()
            alpha = alpha / 2
        iter += 1
        if np.linalg.norm(w - w_old) < tol:
            converged = True
    portfolio_return = w.T @ returns
    portfolio_risk = 0.5 * w.T @ np.diag(risks) @ w
    print("The optimal sparse portfolio consists of the following stocks:")
    cleaned_weights = {}
    for i in range(len(symbols)):
        if w[i] > 0:
            print(f"{symbols[i]}: {w[i]:.2f}")
            cleaned_weights[symbols[i]] = round(w[i], 5)
    print(
        f"The portfolio return is {portfolio_return:.4f} and the portfolio risk is {portfolio_risk:.4f}"
    )
    return cleaned_weights


def calculate_portfolio(df: pd.DataFrame, model: str) -> pd.DataFrame:
    if model == "sparse":
        risk_return_data = calculate_risk_return(df, rf=0)
    else:
        close_data = dp.load_close_data(df)

    match model:
        # Equal Weighted
        case "ew":
            cleaned_weights = {
                symbol: 1 / len(close_data.columns) for symbol in close_data.columns
            }
        # Mean Variance Max Sharpe
        case "mv":
            m = mean_historical_return(close_data)
            s = sample_cov(
                close_data
            )  # s = CovarianceShrinkage(close_df).ledoit_wolf()
            md = EfficientFrontier(m, s)
            weights = md.max_sharpe()
            cleaned_weights = dict(md.clean_weights())
            md.portfolio_performance(verbose=True)
        # Hierarchical Risk Parity
        case "hrp":
            returns = close_data.pct_change().dropna()
            md = HRPOpt(returns)
            weights = md.optimize()
            cleaned_weights = dict(md.clean_weights())
            md.portfolio_performance(verbose=True)
        # mCVAR
        case "mcvar":
            m = mean_historical_return(close_data)
            s = close_data.cov()
            md = EfficientCVaR(m, s)
            weights = md.min_cvar()
            cleaned_weights = dict(md.clean_weights())
            md.portfolio_performance(verbose=True)
        # Sparse
        case "sparse":
            cleaned_weights = sparse_portfolio(risk_return_data)
        case _:
            raise ValueError("Wrong model entered.")

    portfolio_data = dp.get_portfolio_from_dict(dict(cleaned_weights))
    return portfolio_data


def get_portfolio_performance(
    portfolio_data: pd.DataFrame, performance_data: pd.DataFrame, rf: float = 0
) -> tuple[float, float, float]:
    performance_data = performance_data.sort_values("date")
    performance_data = performance_data[
        performance_data["symbol"].isin(portfolio_data["symbol"].unique())
    ]

    returns_df = performance_data.pivot(
        index="date", columns="symbol", values="close"
    ).pct_change()
    weights = portfolio_data.set_index("symbol")["weight"]
    portfolio_returns = (returns_df * weights).sum(axis=1)
    excess_returns = portfolio_returns - rf
    covariance_matrix = returns_df.cov()
    excess_returns_std = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    sharpe_ratio = excess_returns.mean() / excess_returns_std
    return excess_returns.mean(), excess_returns_std, sharpe_ratio
