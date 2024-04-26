from os import mkdir, path

import pandas as pd

from src import data_preparation as dp
from src import model_calculation as mc


def main() -> None:
    END_DATE = "2024-03-01"
    FUTURE_DAYS = 60
    SYMBOLS = 20
    RF = 0

    CLUSTERING_METHODS = ["affinity_propagation", "k_means", "k_medoids"]
    USE_DOMINATION = [True, False]
    SYMBOL_SELECTION_METHODS = ["keep_all", "max_return", "min_risk", "max_sharpe"]
    PORTFOLIO_SELECTION_METHODS = ["ew", "mv", "hrp", "mcvar", "sparse"]

    RESULT_PATH = "./data/result.xlsx"
    COMPACT_RESULT_PATH = "./data/compact_result.xlsx"

    # Load Data

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

    return_df = dp.load_return_data(historical_df)

    # Execute Model
    result_df = pd.DataFrame()
    model_id = 0

    # Clustering Model
    # TODO: Add a KPI to measure how much the model worked
    for clustering_method in CLUSTERING_METHODS:
        clusters_df = mc.load_clusters_data(return_df, model=clustering_method)
        # clusters_number = clusters_df["cluster"].unique().shape[0]
        processed_df = historical_df.merge(clusters_df, on="symbol")
        performance_df = future_df[["symbol", "date", "close", "return"]]
        risk_return_df = mc.calculate_risk_return(processed_df, rf=RF)
        aggregated_df = clusters_df.merge(risk_return_df, on="symbol")

        for use_domination in USE_DOMINATION:

            for symbol_selection_method in SYMBOL_SELECTION_METHODS:
                selected_symbols = mc.select_symbols(
                    aggregated_df,
                    model=symbol_selection_method,
                    dominate=use_domination,
                )
                selected_processed_df = processed_df[
                    processed_df["symbol"].isin(selected_symbols)
                ].copy()

                # Portfolio Selection
                # TODO: CAPM Model, Black-Litterman allocation, etc
                # TODO: long and short? weight_bounds=(-1, 1)
                for portfolio_selection_method in PORTFOLIO_SELECTION_METHODS:
                    portfolio_df = mc.calculate_portfolio(
                        selected_processed_df,
                        model=portfolio_selection_method,
                        sparse_rf=RF,
                    )

                    portfolio_return, portfolio_risk, sharpe_ratio = (
                        mc.get_portfolio_performance(
                            portfolio_df, performance_df, rf=RF
                        )
                    )

                    result_model_df = portfolio_df.merge(clusters_df)
                    result_model_df.insert(0, "model_id", model_id)
                    result_model_df.insert(1, "clustering_method", clustering_method)
                    result_model_df.insert(2, "use_domination", use_domination)
                    result_model_df.insert(
                        3, "symbol_selection_method", symbol_selection_method
                    )
                    result_model_df.insert(
                        4, "portfolio_selection_method", portfolio_selection_method
                    )
                    result_model_df.insert(5, "portfolio_return", portfolio_return)
                    result_model_df.insert(6, "portfolio_risk", portfolio_risk)
                    result_model_df.insert(7, "sharpe_ratio", sharpe_ratio)

                    result_df = pd.concat([result_df, result_model_df])
                    model_id += 1

    compact_result_df = (
        result_df.groupby(by="model_id")
        .first()
        .reset_index()
        .loc[:, "model_id":"sharpe_ratio"]
        .sort_values(by="sharpe_ratio")
    )
    result_df.to_excel(RESULT_PATH, index=False)
    compact_result_df.to_excel(COMPACT_RESULT_PATH, index=False)


if __name__ == "__main__":
    main()
