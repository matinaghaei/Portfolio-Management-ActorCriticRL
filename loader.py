import yfinance as yf
import pandas as pd


class Loader:
    def __init__(self):
        file = open('data/DJIA_tickers_2019.txt', 'r')
        self.tickers = [line.strip() for line in file.readlines()]
        self.stocks = []

    def download_data(self, start_date, end_date=None):
        for ticker in self.tickers:
            print(ticker)
            data = yf.download(ticker, group_by='Ticker', start=start_date, end=end_date)
            data['ticker'] = ticker
            self.stocks.append(data)
            data.to_csv(f'data/ticker_{ticker}.csv')

    def read_data(self):
        for ticker in self.tickers:
            data = pd.read_csv(f'data/ticker_{ticker}.csv', index_col='Date')
            self.stocks.append(data)

    def load(self, start_date=None, end_date=None):
        if start_date is not None:
            self.download_data(start_date, end_date)
        else:
            self.read_data()
        return self.stocks
