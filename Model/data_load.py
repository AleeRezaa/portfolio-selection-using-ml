""" Import Modules and Config """

from cryptocmd import CmcScraper
import pandas as pd
import requests
import json
import warnings

warnings.filterwarnings("ignore")

with open("./config.json", "r") as json_file:
    config = json.load(json_file)


def load_basic_data():
    """Basic Data"""

    # import basic data
    if config["update_data_load"] == False:
        try:
            basic_data = pd.read_csv("./data/basic_data.csv")
        except FileNotFoundError:
            print("Basic data file not found.")
        else:
            return basic_data

    # get basic data
    print("Fetching basic data...")
    url = f'https://api.coinmarketcap.com/data-api/v3/cryptocurrency/listing?start=1&limit={config["number_of_cryptocurrencies"]}&sortBy=market_cap&sortType=desc&convert=USD&cryptoType=all&tagType=all&audited=false'
    response = requests.get(url, timeout=config["timeout"])
    data = json.loads(response.text)
    basic_data = pd.DataFrame(data["data"]["cryptoCurrencyList"])

    # clean basic data
    basic_data = basic_data[
        (basic_data["isActive"] == 1)
        & (basic_data["dateAdded"].apply(lambda x: int(x[:4])) < 2021)
        & (basic_data["lastUpdated"].apply(lambda x: int(x[:4])) > 2022)
        & (basic_data["tags"].apply(lambda x: "stablecoin" not in x))
        & (basic_data["marketPairCount"] > 5)  # TODO: Add Tether
    ]

    # export basic data
    basic_data.reset_index(drop=True, inplace=True)
    basic_data.to_csv("./data/basic_data.csv", index=False)

    print("Basic data file saved.")
    return basic_data


# TODO: delete rows with market cap = 0
def load_historical_data(basic_data):
    """Historical Data"""

    # import historical data
    if config["update_data_load"] == False:
        try:
            historical_data = pd.read_csv("./data/historical_data.csv")
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

    # export historical data
    historical_data.reset_index(drop=True, inplace=True)
    historical_data.to_csv("./data/historical_data.csv", index=False)

    print("Historical data file saved.")
    return historical_data
