from os import mkdir, path

import numpy as np

from src import data_preparation as dp
from src import model_calculation as mc


def main() -> None:
    END_DATE = "2024-03-01"
    FUTURE_DAYS = 60
    SYMBOLS = 20
    RF = 0

    # Data Load

    if not path.exists("data"):
        mkdir("data")

    basic_df = dp.load_basic_data(
        selected_symbols=SYMBOLS,
        max_date_added_year=2020,
        min_last_update_year=2023,
        min_market_pairs=6,
        update=False,
    )

    symbols_list = basic_df["symbol"].to_list()
    historical_df = dp.load_historical_data(symbols_list, update=False)

    historical_df, future_df = dp.filter_historical_data(
        historical_df, selected_days=1460, future_days=FUTURE_DAYS, end_date=END_DATE
    )

    # Clustering Model

    return_df = dp.load_return_data(historical_df)

    # TODO: Add a KPI to measure how much the model worked

    clusters_df = mc.load_clusters_data(return_df, model="affinity_propagation")
    clusters_number = clusters_df["cluster"].unique().shape[0]

    processed_df = historical_df.merge(clusters_df, on="symbol")

    # Portfolio Selection

    # TODO: Use clustering models like HRP
    # TODO: Add CAPM Model
    # TODO: Add Black-Litterman allocation?
    # TODO: long and short? weight_bounds=(-1, 1)
    # TODO: non-covariance models: sparse portfolio, minimum variance portfolio, etc

    risk_return_df = mc.calculate_risk_return(processed_df, rf=RF)
    symbols_df = clusters_df.merge(risk_return_df, on="symbol")

    dominated_df = mc.load_dominated_data(symbols_df)
    portfolio_df = mc.load_ew_portfolio(dominated_df)

    performance_df = future_df[["symbol", "date", "close", "return"]]
    sharpe_ratio = mc.get_portfolio_performance(portfolio_df, performance_df, rf=RF)
    print(f"The Sharpe Ratio for your portfolio is {sharpe_ratio:.2f}.")

    close_df = dp.load_close_data(
        historical_df,
    )

    # Mean Variance Max Sharpe
    portfolio_df = mc.load_portfolio(close_df, "mv")
    sharpe_ratio = mc.get_portfolio_performance(portfolio_df, performance_df, rf=RF)
    print(f"The Sharpe Ratio for your portfolio is {sharpe_ratio:.2f}.")

    # Hierarchical Risk Parity
    portfolio_df = mc.load_portfolio(close_df, "hrp")
    sharpe_ratio = mc.get_portfolio_performance(portfolio_df, performance_df, rf=RF)
    print(f"The Sharpe Ratio for your portfolio is {sharpe_ratio:.2f}.")

    # mCVAR
    portfolio_df = mc.load_portfolio(close_df, "mcvar")
    sharpe_ratio = mc.get_portfolio_performance(portfolio_df, performance_df, rf=RF)
    print(f"The Sharpe Ratio for your portfolio is {sharpe_ratio:.2f}.")

    # sparse portfolio

    # Extract the columns of interest
    symbols = regression_df["symbol"]
    returns = regression_df["future_return"].to_numpy()
    risks = regression_df["future_risk"].to_numpy()

    # Define the portfolio risk function
    def portfolio_risk(w):
        return 0.5 * w.T @ np.diag(risks) @ w

    # Define the portfolio return function
    def portfolio_return(w):
        return w.T @ returns

    # Define the gradient of the portfolio risk function
    def portfolio_risk_grad(w):
        return np.diag(risks) @ w

    # Define the projection onto the simplex
    def project_simplex(w):
        u = np.sort(w)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1.0)
        return np.maximum(w - theta, 0)

    # Define the thresholding operator
    def threshold(w, k):
        idx = np.argsort(w)[::-1][:k]
        w_new = np.zeros_like(w)
        w_new[idx] = w[idx]
        return w_new

    # Set the number of non-zero weights (e.g. 10)
    k = 10
    # Set the target portfolio return (e.g. 0.05)
    r = 0.001
    # Set the initial portfolio weights (e.g. uniform)
    w = np.ones(len(symbols)) / len(symbols)
    # Set the step size for gradient descent (e.g. 0.01)
    alpha = 0.01
    # Set the tolerance for convergence (e.g. 1e-6)
    tol = 1e-6
    # Set the maximum number of iterations (e.g. 1000)
    max_iter = 10000
    # Initialize the iteration counter and the convergence flag
    iter = 0
    converged = False
    # Run the algorithm until convergence or maximum iterations
    while not converged and iter < max_iter:
        # Save the current weights
        w_old = w.copy()
        # Update the weights using gradient descent
        w = w - alpha * portfolio_risk_grad(w)
        # Project the weights onto the simplex
        w = project_simplex(w)
        # Threshold the weights to enforce sparsity
        w = threshold(w, k)
        # Check the portfolio return constraint
        if portfolio_return(w) < r:
            # If violated, reset the weights to the previous ones
            w = w_old.copy()
            # Reduce the step size
            alpha = alpha / 2
        # Increment the iteration counter
        iter += 1
        # Check the convergence criterion
        if np.linalg.norm(w - w_old) < tol:
            # If satisfied, set the convergence flag to True
            converged = True
    # Get the portfolio return and risk
    portfolio_return = portfolio_return(w)
    portfolio_risk = portfolio_risk(w)
    # Print the results
    print("The optimal sparse portfolio consists of the following stocks:")
    cleaned_weights = {}
    for i in range(len(symbols)):
        if w[i] > 0:
            print(f"{symbols[i]}: {w[i]:.2f}")
            cleaned_weights[symbols[i]] = round(w[i], 5)
    print(
        f"The portfolio return is {portfolio_return:.4f} and the portfolio risk is {portfolio_risk:.4f}"
    )

    portfolio_df = dp.get_portfolio_from_dict(dict(cleaned_weights))
    sharpe_ratio = mc.get_portfolio_performance(portfolio_df, performance_df, rf=RF)
    print(f"The Sharpe Ratio for your portfolio is {sharpe_ratio:.2f}.")


if __name__ == "__main__":
    main()
