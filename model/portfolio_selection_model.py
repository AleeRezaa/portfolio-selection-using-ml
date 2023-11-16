from itertools import combinations
import pandas as pd

DOMINATED_DATA_PATH = "./data/dominated_data.csv"


def load_dominated_data(prediction_data, update=False):
    """Dominated Data"""

    # import dominated data
    if update == False:
        try:
            dominated_data = pd.read_csv(DOMINATED_DATA_PATH)
        except FileNotFoundError:
            print("Dominated data file not found.")
        else:
            return dominated_data

    dominated_data = prediction_data[
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

    # export dominated data
    dominated_data.to_csv(DOMINATED_DATA_PATH, index=False)
    return dominated_data
