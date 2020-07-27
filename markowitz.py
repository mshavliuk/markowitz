from typing import Optional

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime
import os.path
import pandas
import scipy.optimize as optimization
from pandas import DataFrame, Series
import requests
import json

work_days = 252

start_date = datetime.datetime(2015, 1, 1)
end_date = datetime.datetime(2020, 7, 1)


def convert_isin_to_tickers(isin):
    query_template = 'https://query1.finance.yahoo.com/v1/finance/search?lang=en-US&newsCount=0&q={}'
    search_url = query_template.format(isin)
    r = requests.get(search_url)
    if r.status_code != 200:
        return []
    quotes = list(map(lambda q: q['symbol'], r.json()['quotes']))
    return quotes


def get_isin_to_ticker_dict(isins):
    filename = './.cache/isin_to_tracker.json'
    if os.path.isfile(filename):
        with open(filename) as json_file:
            isin_to_ticker = json.load(json_file)
    else:
        isin_to_ticker = {}

    changed = False
    for isin in isins:
        if isin not in isin_to_ticker:
            changed = True
            isin_to_ticker[isin] = convert_isin_to_tickers(isin)

    if changed:
        with open(filename, 'w') as outfile:
            json.dump(isin_to_ticker, outfile)
    return isin_to_ticker


def download_with_yahoo(ticker, start, end) -> Series:
    try:
        ticker_data = web.DataReader(ticker, 'yahoo', start, end)
    except:
        return Series(dtype='float64')
    return ticker_data['Adj Close']


def download_with_av(ticker, start, end) -> Series:
    try:
        ticker_data = web.DataReader(ticker, 'av-daily', start, end)
    except:
        return Series(dtype='float64')

    if ticker_data.empty:
        return Series(dtype='float64')

    close_data = ticker_data['close']
    close_data.index = pd.to_datetime(close_data.index)
    return close_data


def download_stock(ticker, start, end) -> Optional[Series]:
    filename = './.cache/' + ticker + '.json'
    if os.path.isfile(filename):
        with open(filename) as json_file:
            ticker_data = pandas.read_json(json_file, typ='series')
            # TODO: exclude the extra parts or load if needed
    else:
        av_ticker_data = download_with_av(ticker, start, end)
        ya_ticker_data = download_with_yahoo(ticker, start, end)

        differences = abs((av_ticker_data / ya_ticker_data) - 1)
        if (differences > 0.1).any():
            print('something is wrong here - big differences')
        df = DataFrame((av_ticker_data, ya_ticker_data))
        ticker_data = df.mean()

    if ticker_data.empty:
        return None

    fluctuations = abs((ticker_data / ticker_data.shift(1)) - 1)
    if (fluctuations > .2).any():
        print('something is wrong here - big fluctuations')
        return None

    if ticker_data.count() / pd.date_range(start, end).value_counts().count() < .5:
        print('to small data')
        return None

    filtered_data = filter_stock_data(ticker_data)
    filtered_data.to_json(filename)
    return filtered_data


def filter_stock_data(stock_data: Series) -> Series:
    fluctuations = abs((stock_data / stock_data.shift(1)) - 1) < .1

    result = stock_data[fluctuations]
    return result


def download_data(stocks):
    index = pd.date_range(start_date, end_date)
    data = DataFrame(index=index)
    for ticker in stocks:
        ticker_data = download_stock(ticker, start_date, end_date)
        if ticker_data is not None:
            data[ticker] = ticker_data

    if data.empty:
        print('not found')
        return None

    data['mean'] = data.mean(axis=1)
    # data.plot(title=','.join(data.columns))
    # plt.show()
    return data


def show_data(data):
    data.plot()
    plt.show()


def calculate_returns(data):
    changes = data / data.shift(1)

    returns = np.log(changes)
    return returns


def plot_daily_returns(returns):
    returns.plot()
    plt.show()


def show_statistics(returns):
    print(returns.mean() * work_days)
    print(returns.cov() * work_days)


def initialize_weights(number):
    weights = np.random.random(number)
    weights /= np.sum(weights)
    return weights


def calculate_portfolio_return(returns, weights):
    return np.sum(returns.mean() * weights) * work_days
    # print('Expected portfolio return:', portfolio_return)


def calculate_portfolio_variance(returns, weights):
    return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * work_days, weights)))
    # print('Expected variance:', portfolio_variance)


def generate_portfolios(returns):
    preturns = []
    pvariances = []
    pweights = []

    for i in range(10_000):
        weights = initialize_weights(len(returns))
        preturns.append(calculate_portfolio_return(returns, weights))
        pvariances.append(calculate_portfolio_variance(returns, weights))
        pweights.append(weights)

    return np.array(preturns), np.array(pvariances), np.array(pweights)


def plot_portfolios(returns, variances):
    plt.figure()
    plt.scatter(variances, returns, c=returns / variances, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()


def statistics(weights, returns):
    weights = np.array(weights)
    portfolio_return = calculate_portfolio_return(returns, weights)
    portfolio_volatility = calculate_portfolio_variance(returns, weights)
    return np.array(
        [portfolio_return, portfolio_volatility, portfolio_return / portfolio_volatility])


def min_func_sharpe(weights, returns):
    return -statistics(weights, returns)[2]


def optimize_portfolio(weights, returns, first_ticker_weight):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # bounds = list((0, 1) for x in range(len(weights)))

    bounds = list((0, 1) for x in range(len(returns.columns)))

    ### increase preferable ticker bound
    bounds[0] = (first_ticker_weight, 1)

    optimum = optimization.minimize(fun=min_func_sharpe, x0=weights, args=returns, method='SLSQP',
                                    bounds=bounds, constraints=constraints)
    return Series(optimum['x'], index=returns.columns)


def print_optimal_portfolio(optimum, returns):
    print('Optimal weights:', optimum.round(3)[optimum > 0.01])
    expectations = statistics(optimum, returns)
    print('Expected return volatility and sharpe ratio:', expectations)
    # print('Expected volatility:')
    # print('Expected sharpe ratio:')


def show_optimal_portfolio(optimum, returns, preturns, pvariances):
    plt.figure()
    plt.scatter(pvariances, preturns, c=preturns / pvariances, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    expectations = statistics(optimum['x'].round(3), returns)
    plt.plot(expectations[1], expectations[0], 'g*', markersize=20.0)
    plt.show()


if __name__ == '__main__':
    with open('./selected_isins.json') as json_file:
        isins = json.load(json_file)

    isins_to_tickers = get_isin_to_ticker_dict(isins)
    data = DataFrame()
    data_to_show = DataFrame()
    counter = 0
    for isin in isins:
        counter += 1
        if len(isins_to_tickers[isin]) == 0:
            continue
        tickers_data = download_data(isins_to_tickers[isin])
        if tickers_data is not None:
            data[isin] = tickers_data['mean']
            print(isin, '[{}/{}]'.format(counter, len(isins_to_tickers)))
        # data_to_show[isin] = data[isin] / data[isin][data[isin].first_valid_index()]
        # data_to_show.plot(title=isin)
        # plt.show()

    # show_data(data)
    returns = calculate_returns(data)
    # plot_daily_returns(returns)
    show_statistics(returns)
    # preturns, pvariances, pweights = generate_portfolios(returns)
    # plot_portfolios(preturns, pvariances)
    weights = initialize_weights(len(data.columns))
    # for w in np.arange(0.1, 1, 0.1):
    #     optimum = optimize_portfolio(weights, returns, w)
    #     print_optimal_portfolio(optimum, returns)

    optimum = optimize_portfolio(weights, returns, 0)
    print_optimal_portfolio(optimum, returns)
    # show_optimal_portfolio(optimum, returns, preturns, pvariances)
