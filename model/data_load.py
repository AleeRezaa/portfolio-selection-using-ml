import json
import warnings
from time import sleep

import pandas as pd
import requests
from cryptocmd import CmcScraper

warnings.filterwarnings("ignore")

BASIC_DATA_URL = "https://api.coinmarketcap.com/data-api/v3/cryptocurrency/listing?start=1&sortBy=market_cap&sortType=desc&convert=USD&cryptoType=all&tagType=all&audited=false&limit="
BASIC_DATA_PATH = "./data/basic_data.csv"
HISTORICAL_DATA_PATH = "./data/historical_data.csv"
SELECTED_DATA_PATH = "./data/selected_data.csv"
TIMEOUT = 10
TRIES = 5
SLEEP = 5


def load_basic_data(
    selected_symbols=100,
    max_date_added_year=2020,
    min_last_update_year=2023,
    min_market_pairs=6,
    update=False,
) -> pd.DataFrame:
    """Basic Data"""

    # import basic data
    if update == False:
        try:
            basic_data = pd.read_csv(BASIC_DATA_PATH)
        except FileNotFoundError:
            print("Basic data file not found.")
        else:
            return basic_data

    # get basic data
    print("Fetching basic data...")
    url = BASIC_DATA_URL + str(selected_symbols)
    response = requests.get(url, timeout=TIMEOUT)
    data = json.loads(response.text)
    basic_data = pd.DataFrame(data["data"]["cryptoCurrencyList"])

    # clean basic data
    basic_data = basic_data[
        (basic_data["isActive"] == 1)
        & (basic_data["dateAdded"].apply(lambda x: int(x[:4])) <= max_date_added_year)
        & (
            basic_data["lastUpdated"].apply(lambda x: int(x[:4]))
            >= min_last_update_year
        )
        # & (basic_data["tags"].apply(lambda x: "stablecoin" not in x))
        & (basic_data["marketPairCount"] >= min_market_pairs)
    ]

    # export basic data
    basic_data.reset_index(drop=True, inplace=True)
    basic_data.to_csv(BASIC_DATA_PATH, index=False)

    print("Basic data file saved.")
    return basic_data


def load_historical_data(symbols_list, update=False) -> pd.DataFrame:
    """Historical Data"""

    # import historical data
    if update == False:
        try:
            historical_data = pd.read_csv(HISTORICAL_DATA_PATH)
        except FileNotFoundError:
            print("Historical data file not found.")
        else:
            return historical_data

    historical_data = pd.DataFrame()
    n = 0
    number_of_symbols = len(symbols_list)

    # loop in symbols and get historical data
    for symbol in symbols_list:
        n += 1
        if "," in symbol:
            symbol = symbol.split(",")[0]
        scraper = CmcScraper(symbol)

        t = 0
        while t < TRIES:
            try:
                symbol_historical_data = scraper.get_dataframe()
            except ConnectionError:
                print(f"Connection error for {symbol}, trying again...")
                t += 1
            except Exception as e:
                print(
                    f"{n}/{number_of_symbols} Error in fetching historical data for {symbol}: {e}"
                )
                successful_fetch = False
                break
            else:
                print(
                    f"{n}/{number_of_symbols} Historical data for {symbol} successfully fetched."
                )
                successful_fetch = True
                break
        else:
            print(
                f"{n}/{number_of_symbols} Connection error in fetching historical data for {symbol} after {TRIES} tries."
            )
            successful_fetch = False

        if successful_fetch:
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

    historical_data = historical_data[
        (historical_data["marketcap"] > 0) & (historical_data["volume"] > 0)
    ]

    # export historical data
    historical_data.reset_index(drop=True, inplace=True)
    historical_data.to_csv(HISTORICAL_DATA_PATH, index=False)

    print("Historical data file saved.")
    return historical_data


def load_selected_data(
    historical_data, selected_days=1460, end_date=None, update=False
) -> pd.DataFrame:
    """Selected Data"""

    # import selected data
    if update == False:
        try:
            selected_data = pd.read_csv(SELECTED_DATA_PATH)
        except FileNotFoundError:
            print("Selected data file not found.")
        else:
            return selected_data

    if end_date is not None:
        historical_data = historical_data[historical_data["date"] <= end_date]

    symbols_age = historical_data.groupby("symbol").count()["return"]

    # keep symbols which have at least n days history of return data
    selected_data = historical_data[
        historical_data["symbol"].isin(dict(symbols_age[symbols_age >= selected_days]))
    ]
    print(
        f"delete symbols which do not have at least {selected_days} days of return data: {[x for x in dict(symbols_age[symbols_age < selected_days]).keys()]}"
    )

    # keep the last n days history of return data
    selected_data = selected_data.groupby("symbol").head(selected_days)

    # keep symbols which have the last date of return data
    symbols_last_date = selected_data.groupby("symbol").first()["date"]
    last_date = symbols_last_date["BTC"]
    selected_data = selected_data[
        selected_data["symbol"].isin(
            dict(symbols_last_date[symbols_last_date == last_date])
        )
    ]
    print(
        f"delete symbols which do not have the last day of return data: {[x for x in dict(symbols_last_date[symbols_last_date != last_date]).keys()]}"
    )

    # keep symbols which have the first date of return data
    symbols_first_date = selected_data.groupby("symbol").last()["date"]
    first_date = symbols_first_date["BTC"]
    selected_data = selected_data[
        selected_data["symbol"].isin(
            dict(symbols_first_date[symbols_first_date == first_date])
        )
    ]
    print(
        f"delete symbols which do not have the first day of return data: {[x for x in dict(symbols_first_date[symbols_first_date != first_date]).keys()]}"
    )

    # export selected data
    selected_data.reset_index(drop=True, inplace=True)
    selected_data.to_csv(SELECTED_DATA_PATH, index=False)

    print("Selected data file saved.")
    return selected_data
