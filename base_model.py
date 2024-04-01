from os import mkdir, path

from src import data_preparation as dp
from src import model_calculation as mc


def main() -> None:
    END_DATE = "2024-03-01"
    FUTURE_DAYS = 60
    SYMBOLS = 20
    RF = 0

    CLUSTERING_METHODS = ["affinity_propagation"]
    USE_DOMINATION = [True, False]
    SYMBOL_SELECTION_METHODS = ["keep_all", "max_return", "min_risk", "max_sharpe"]
    PORTFOLIO_SELECTION_METHODS = ["ew", "mv", "hrp", "mcvar", "sparse"]

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

    # Clustering Model
    # TODO: Add a KPI to measure how much the model worked
    for clustering_method in CLUSTERING_METHODS:
        print(f"🔵🔵🔵🔵 clustering_method: {clustering_method}")

        clusters_df = mc.load_clusters_data(return_df, model=clustering_method)
        clusters_number = clusters_df["cluster"].unique().shape[0]

        processed_df = historical_df.merge(clusters_df, on="symbol")

        performance_df = future_df[["symbol", "date", "close", "return"]]

        risk_return_df = mc.calculate_risk_return(processed_df, rf=RF)
        aggregated_df = clusters_df.merge(risk_return_df, on="symbol")

        for use_domination in USE_DOMINATION:
            print(f"🔵🔵🔵 use_domination: {use_domination}")

            for symbol_selection_method in SYMBOL_SELECTION_METHODS:
                print(f"🔵🔵 symbol_selection_method: {symbol_selection_method}")
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
                    print(
                        f"🔵 portfolio_selection_method: {portfolio_selection_method}"
                    )

                    portfolio_df = mc.calculate_portfolio(
                        selected_processed_df, portfolio_selection_method
                    )
                    sharpe_ratio = mc.get_portfolio_performance(
                        portfolio_df, performance_df, rf=RF
                    )
                    print(
                        f"🟢 The Sharpe Ratio for your portfolio is {sharpe_ratio:.2f}."
                    )


if __name__ == "__main__":
    main()
