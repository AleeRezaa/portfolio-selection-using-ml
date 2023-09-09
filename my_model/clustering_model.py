import warnings

import numpy as np
import pandas as pd
from sklearn import cluster, covariance

warnings.filterwarnings("ignore")

RETURN_DATA_PATH = "./data/return_data.csv"
CLUSTERS_DATA_PATH = "./data/clusters_data.csv"


def load_return_data(selected_data, update=False):
    """Return Data"""

    # import return data
    if update == False:
        try:
            return_data = pd.read_csv(RETURN_DATA_PATH)
        except FileNotFoundError:
            print("Return data file not found.")
        else:
            return return_data

    return_data = selected_data[["date", "symbol", "return"]]

    # create the pivot table of return data
    return_data = return_data.pivot(index="date", columns="symbol")
    return_data = return_data["return"]

    return_data.index.name = None
    return_data.columns.name = None

    # export return data
    return_data.to_csv(RETURN_DATA_PATH, index=False)

    print("return data file saved.")
    return return_data


# https://www.relataly.com/crypto-market-cluster-analysis-using-affinity-propagation-python/8114/
# Affinity Propagation انتشار وابستگی
def load_clusters_data(return_data, update):
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
