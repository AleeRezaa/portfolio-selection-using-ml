from os import mkdir, path

import pandas as pd

from src import data_preparation as dp
from src import model_calculation as mc


def main() -> None:
    HISTORY_DAYS = 360
    FUTURE_DAYS = 60
    SYMBOLS = 40
    RF = 0
    UPDATE_HISTORY = False

    CLUSTERING_METHODS = ["affinity_propagation", "k_means", "k_medoids"]
    USE_DOMINATION = [True, False]
    SYMBOL_SELECTION_METHODS = ["max_return", "min_risk", "max_sharpe", "keep_all"]
    PORTFOLIO_SELECTION_METHODS = ["ew", "mv", "mcvar", "hrp", "sparse"]
    END_DATES = [
        str(x)[:10]
        for x in pd.date_range(start="2021-01-01", end="2024-03-01", freq="2MS")
    ]

    RESULT_PATH = "./data/result.xlsx"
    COMPACT_RESULT_PATH = "./data/compact_result.xlsx"

    result_df = pd.DataFrame()
    id = 0

    # Load Data

    if not path.exists("data"):
        mkdir("data")

    basic_df = dp.load_basic_data(
        selected_symbols=SYMBOLS,
        max_date_added_year=2018,
        min_last_update_year=2024,
        min_market_pairs=6,
        update=UPDATE_HISTORY,
    )

    symbols_list = basic_df["symbol"].to_list()
    historical_df = dp.load_historical_data(symbols_list, update=UPDATE_HISTORY)

    for end_date in END_DATES:
        history_df, future_df = dp.filter_historical_data(
            historical_df,
            selected_days=HISTORY_DAYS,
            future_days=FUTURE_DAYS,
            end_date=end_date,
        )

        return_df = dp.load_return_data(history_df)

        # Clustering Model
        # TODO: Add a KPI to measure how much the model worked
        for clustering_method in CLUSTERING_METHODS:
            clusters_df = mc.load_clusters_data(return_df, model=clustering_method)
            # clusters_number = clusters_df["cluster"].unique().shape[0]
            processed_df = history_df.merge(clusters_df, on="symbol")
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
                        model_name = f"{clustering_method}/{use_domination}/{symbol_selection_method}/{portfolio_selection_method}"
                        print(f"\n\n{end_date}/{model_name}")

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
                        result_model_df.insert(0, "id", id)
                        result_model_df.insert(1, "end_date", end_date)
                        result_model_df.insert(2, "model_name", model_name)
                        result_model_df.insert(
                            3, "clustering_method", clustering_method
                        )
                        result_model_df.insert(4, "use_domination", use_domination)
                        result_model_df.insert(
                            5, "symbol_selection_method", symbol_selection_method
                        )
                        result_model_df.insert(
                            6, "portfolio_selection_method", portfolio_selection_method
                        )
                        result_model_df.insert(7, "portfolio_return", portfolio_return)
                        result_model_df.insert(8, "portfolio_risk", portfolio_risk)
                        result_model_df.insert(9, "sharpe_ratio", sharpe_ratio)

                        result_df = pd.concat([result_df, result_model_df])
                        id += 1

        result_df.to_excel(RESULT_PATH, index=False)

        compact_result_df = (
            result_df.groupby(by="id")
            .first()
            .reset_index()
            .loc[:, "model_name":"sharpe_ratio"]
            .groupby(
                by=[
                    "model_name",
                    "clustering_method",
                    "use_domination",
                    "symbol_selection_method",
                    "portfolio_selection_method",
                ]
            )
            .mean()
            .reset_index()
            .sort_values(by="sharpe_ratio", ascending=False)
        )
        compact_result_df.to_excel(COMPACT_RESULT_PATH, index=False)


if __name__ == "__main__":
    main()
