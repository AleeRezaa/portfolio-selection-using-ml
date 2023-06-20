""" Import Modules and Config """

import pandas as pd 
import numpy as np 
from sklearn import cluster, covariance
import json
import warnings

warnings.filterwarnings('ignore')

with open('./config.json', 'r') as json_file:
    config = json.load(json_file)


def load_changes_data(historical_data):
    """ Changes Data """

    # import changes data
    if config['update_changes_data'] == False:
        try:
            changes_data = pd.read_csv('./data/changes_data.csv', index_col = 'Unnamed: 0')
        except FileNotFoundError:
            print('Changes data file not found.')
        else:
            return changes_data

    changes_data = historical_data[['date', 'symbol', 'change']]
    symbols_age = changes_data.groupby('symbol').count()['change']
    
    # keep symbols which have at least two years of changes data
    changes_data = changes_data[historical_data['symbol'].isin(dict(symbols_age[symbols_age >= config['symbols_minimum_age']]))]
    print(f'delete symbols which do not have at least {config["symbols_minimum_age"]} days of changes data: {[x for x in dict(symbols_age[symbols_age < config["symbols_minimum_age"]]).keys()]}')

    # keep the last two years of changes data
    changes_data = changes_data.groupby('symbol').head(config['symbols_minimum_age'])

    # keep symbols which have the last date of changes data
    symbols_last_date = changes_data.groupby('symbol').first()['date']
    last_date = symbols_last_date['BTC']
    changes_data = changes_data[historical_data['symbol'].isin(dict(symbols_last_date[symbols_last_date == last_date]))]
    print(f'delete symbols which do not have the last day of changes data: {[x for x in dict(symbols_last_date[symbols_last_date != last_date]).keys()]}')

    # keep symbols which have the first date of changes data
    symbols_first_date = changes_data.groupby('symbol').last()['date']
    first_date = symbols_first_date['BTC']
    changes_data = changes_data[historical_data['symbol'].isin(dict(symbols_first_date[symbols_first_date == first_date]))]
    print(f'delete symbols which do not have the first day of changes data: {[x for x in dict(symbols_first_date[symbols_first_date != first_date]).keys()]}')
    
    #create the pivot table of changes data
    changes_data = changes_data.pivot(index = 'date', columns = 'symbol')
    changes_data = changes_data['change']
    
    changes_data.index.name = None
    changes_data.columns.name = None

    # export changes data
    changes_data.to_csv('./data/changes_data.csv')

    print('Changes data file saved.')
    return changes_data


def load_clusters(changes_data):
    """ Clusters """
    
    # import clusters
    if config['update_clusters'] == False:
        try:
            with open('./data/clusters.json', 'r') as json_file:
                clusters = json.load(json_file)
        except FileNotFoundError:
            print('Clusters file not found.')
        else:
            return clusters

    # create clustering data
    symbols = changes_data.columns
    clustering_data = np.array(changes_data / changes_data.std())

    # create clustering modol
    edge_model = covariance.GraphicalLassoCV()
    edge_model.fit(clustering_data)
    cluster_centers_indices, labels = cluster.affinity_propagation(edge_model.covariance_, random_state = 1)
    n = labels.max() + 1

    # create clusters
    clusters = []
    for i in range(n):
        sub_cluster = list(symbols[labels == i])
        clusters.append(sub_cluster)

    # export clusters
    with open('./data/clusters.json', 'w') as json_file:
        json.dump(clusters, json_file, indent = 4)

    print('Clusters file saved.')
    return clusters