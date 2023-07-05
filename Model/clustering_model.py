""" Import Modules and Config """

import pandas as pd
import numpy as np
from sklearn import cluster, covariance
import json
import warnings

warnings.filterwarnings("ignore")

with open("./config.json", "r") as json_file:
    config = json.load(json_file)


# TODO: add to_date
def load_return_data(historical_data):
    """Return Data"""

    # import return data
    if config["update_clustering_model"] == False:
        try:
            return_data = pd.read_csv("./data/return_data.csv")
        except FileNotFoundError:
            print("Return data file not found.")
        else:
            return return_data

    return_data = historical_data[["date", "symbol", "return"]]
    symbols_age = return_data.groupby("symbol").count()["return"]

    # keep symbols which have at least two years of return data
    return_data = return_data[
        historical_data["symbol"].isin(
            dict(symbols_age[symbols_age >= config["symbols_minimum_age"]])
        )
    ]
    print(
        f'delete symbols which do not have at least {config["symbols_minimum_age"]} days of return data: {[x for x in dict(symbols_age[symbols_age < config["symbols_minimum_age"]]).keys()]}'
    )

    # keep the last two years of return data
    return_data = return_data.groupby("symbol").head(config["symbols_minimum_age"])

    # keep symbols which have the last date of return data
    symbols_last_date = return_data.groupby("symbol").first()["date"]
    last_date = symbols_last_date["BTC"]
    return_data = return_data[
        historical_data["symbol"].isin(
            dict(symbols_last_date[symbols_last_date == last_date])
        )
    ]
    print(
        f"delete symbols which do not have the last day of return data: {[x for x in dict(symbols_last_date[symbols_last_date != last_date]).keys()]}"
    )

    # keep symbols which have the first date of return data
    symbols_first_date = return_data.groupby("symbol").last()["date"]
    first_date = symbols_first_date["BTC"]
    return_data = return_data[
        historical_data["symbol"].isin(
            dict(symbols_first_date[symbols_first_date == first_date])
        )
    ]
    print(
        f"delete symbols which do not have the first day of return data: {[x for x in dict(symbols_first_date[symbols_first_date != first_date]).keys()]}"
    )

    # create the pivot table of return data
    return_data = return_data.pivot(index="date", columns="symbol")
    return_data = return_data["return"]

    return_data.index.name = None
    return_data.columns.name = None

    # export return data
    return_data.to_csv("./data/return_data.csv", index=False)

    print("return data file saved.")
    return return_data


def load_clusters_data(return_data):
    """Clusters Data"""

    # import clusters data
    if config["update_clustering_model"] == False:
        try:
            clusters_data = pd.read_csv("./data/clusters_data.csv")
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
            clusters_data.loc[len(clusters_data)] = [symbol, str(c)]

    # export clusters data
    clusters_data.to_csv("./data/clusters_data.csv", index=False)

    print("Clusters data file saved.")
    return clusters_data
