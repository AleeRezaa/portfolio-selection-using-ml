""" Import Modules """

import json
import warnings

import pandas as pd
import requests
from cryptocmd import CmcScraper

warnings.filterwarnings("ignore")

BASIC_DATA_PATH = "./data/basic_data.csv"
HISTORICAL_DATA_PATH = "./data/historical_data.csv"
TIMEOUT = 10


def load_basic_data(update_data_load, number_of_cryptocurrencies):
    """Basic Data"""

    # import basic data
    if update_data_load == False:
        try:
            basic_data = pd.read_csv(BASIC_DATA_PATH)
        except FileNotFoundError:
            print("Basic data file not found.")
        else:
            return basic_data

    # get basic data
    print("Fetching basic data...")
    url = f"https://api.coinmarketcap.com/data-api/v3/cryptocurrency/listing?start=1&limit={number_of_cryptocurrencies}&sortBy=market_cap&sortType=desc&convert=USD&cryptoType=all&tagType=all&audited=false"
    response = requests.get(url, timeout=TIMEOUT)
    data = json.loads(response.text)
    basic_data = pd.DataFrame(data["data"]["cryptoCurrencyList"])

    # clean basic data
    basic_data = basic_data[
        (basic_data["isActive"] == 1)
        & (basic_data["dateAdded"].apply(lambda x: int(x[:4])) < 2021)
        & (basic_data["lastUpdated"].apply(lambda x: int(x[:4])) > 2022)
        # & (basic_data["tags"].apply(lambda x: "stablecoin" not in x))
        & (basic_data["marketPairCount"] > 5)
    ]

    # export basic data
    basic_data.reset_index(drop=True, inplace=True)
    basic_data.to_csv(BASIC_DATA_PATH, index=False)

    print("Basic data file saved.")
    return basic_data


def load_historical_data(basic_data, update_data_load, history_days):
    """Historical Data"""

    # import historical data
    if update_data_load == False:
        try:
            historical_data = pd.read_csv(HISTORICAL_DATA_PATH)
        except FileNotFoundError:
            print("Historical data file not found.")
        else:
            return historical_data

    # create symbols list and historical data dataframe
    symbols = basic_data["symbol"].to_list()
    historical_data = pd.DataFrame()
    n = 0
    number_of_symbols = len(symbols)

    # loop in symbols and get historical data
    for symbol in symbols:
        if "," in symbol:
            symbol = symbol.split(",")[0]

        n += 1
        print(f"Fetching historical data for {symbol} ({n}/{number_of_symbols})")
        scraper = CmcScraper(symbol)

        try:
            symbol_historical_data = scraper.get_dataframe()
        except:
            print(f"Error in fetching historical data for {symbol}")
            continue

        symbol_historical_data.columns = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "marketcap",
        ]
        symbol_historical_data.insert(0, "symbol", symbol)
        symbol_historical_data.insert(
            6,
            "avg",
            (symbol_historical_data["open"] + symbol_historical_data["close"]) / 2,
        )
        symbol_historical_data.insert(
            7,
            "return",
            (symbol_historical_data["close"] - symbol_historical_data["open"])
            / symbol_historical_data["open"],
        )

        historical_data = pd.concat([historical_data, symbol_historical_data])

    symbols_age = historical_data.groupby("symbol").count()["return"]

    # keep symbols which have at least n days history of return data
    historical_data = historical_data[
        historical_data["symbol"].isin(dict(symbols_age[symbols_age >= history_days]))
    ]
    print(
        f"delete symbols which do not have at least {history_days} days of return data: {[x for x in dict(symbols_age[symbols_age < history_days]).keys()]}"
    )

    # keep the last n days history of return data
    historical_data = historical_data.groupby("symbol").head(history_days)

    # keep symbols which have the last date of return data
    symbols_last_date = historical_data.groupby("symbol").first()["date"]
    last_date = symbols_last_date["BTC"]
    historical_data = historical_data[
        historical_data["symbol"].isin(
            dict(symbols_last_date[symbols_last_date == last_date])
        )
    ]
    print(
        f"delete symbols which do not have the last day of return data: {[x for x in dict(symbols_last_date[symbols_last_date != last_date]).keys()]}"
    )

    # keep symbols which have the first date of return data
    symbols_first_date = historical_data.groupby("symbol").last()["date"]
    first_date = symbols_first_date["BTC"]
    historical_data = historical_data[
        historical_data["symbol"].isin(
            dict(symbols_first_date[symbols_first_date == first_date])
        )
    ]
    print(
        f"delete symbols which do not have the first day of return data: {[x for x in dict(symbols_first_date[symbols_first_date != first_date]).keys()]}"
    )

    # export historical data
    historical_data.reset_index(drop=True, inplace=True)
    historical_data.to_csv(HISTORICAL_DATA_PATH, index=False)

    print("Historical data file saved.")
    return historical_data
