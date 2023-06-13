""" Import Modules and Config """

from cryptocmd import CmcScraper
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns
from sklearn import cluster, covariance, manifold
import requests
import json
import warnings
warnings.filterwarnings('ignore')

with open('./config.json', 'r') as json_file:
    config = json.load(json_file)



""" Basic Data """

def load_basic_data():

    # import basic data
    if config['update_basic_data'] == False:
        try:
            basic_data = pd.read_csv('./data/basic_data.csv', index_col = 'Unnamed: 0')
        except FileNotFoundError:
            print('Basic data file not found.')
        else:
            return basic_data

    # get basic data
    print('Fetching basic data...')
    url = f'https://api.coinmarketcap.com/data-api/v3/cryptocurrency/listing?start=1&limit={config["number_of_cryptocurrencies"]}&sortBy=market_cap&sortType=desc&convert=USD&cryptoType=all&tagType=all&audited=false'
    response = requests.get(url, timeout = config['timeout'])
    data = json.loads(response.text)
    basic_data = pd.DataFrame(data['data']['cryptoCurrencyList'])

    # clean basic data
    basic_data = basic_data[
        (basic_data['isActive'] == 1) & 
        (basic_data['dateAdded'].apply(lambda x: int(x[:4])) < 2021) & 
        (basic_data['lastUpdated'].apply(lambda x: int(x[:4])) > 2022) &
        (basic_data['tags'].apply(lambda x: 'stablecoin' not in x)) &  # TODO: Add Tether
        (basic_data['marketPairCount'] > 5)
    ]

    # export basic data
    basic_data.reset_index(drop = True, inplace = True)
    basic_data.to_csv('./data/basic_data.csv')

    print('Basic data file saved.')
    return basic_data



""" Historical Data """

def load_historical_data(basic_data):

    # import historical data
    if config['update_historical_data'] == False:
        try:
            historical_data = pd.read_csv('./data/historical_data.csv', index_col = 'Unnamed: 0')
        except FileNotFoundError:
            print('Historical data file not found.')
        else:
            return historical_data

    # create symbols list and historical data dataframe
    symbols = basic_data['symbol'].to_list()
    historical_data = pd.DataFrame()
    n = 0
    number_of_symbols = len(symbols)

    # loop in symbols and get historical data
    for symbol in symbols:

        if ',' in symbol:
            symbol = symbol.split(',')[0]

        n += 1
        print(f'Fetching historical data for {symbol} ({n}/{number_of_symbols})')
        scraper = CmcScraper(symbol)

        try:
            symbol_historical_data = scraper.get_dataframe()
        except:
            print(f'Error in fetching historical data for {symbol}')
            continue

        symbol_historical_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'marketcap']
        symbol_historical_data.insert(0, 'symbol', symbol)
        symbol_historical_data.insert(6, 'avg', (symbol_historical_data['open'] + symbol_historical_data['close']) / 2)
        symbol_historical_data.insert(7, 'change', (symbol_historical_data['close'] - symbol_historical_data['open']) / symbol_historical_data['open'])

        historical_data = pd.concat([historical_data, symbol_historical_data])

    # export historical data
    historical_data.reset_index(drop = True, inplace = True)
    historical_data.to_csv('./data/historical_data.csv')

    print('Historical data file saved.')
    return historical_data



""" Changes Data """

def load_changes_data(historical_data):

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



""" Clusters """

def load_clusters(changes_data):
    
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