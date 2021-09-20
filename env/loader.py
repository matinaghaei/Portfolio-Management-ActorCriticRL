import yfinance as yf
import pandas as pd


class Loader:
    def __init__(self):
        file = open('env/data/DJIA_tickers_2019.txt', 'r')
        self.tickers = [line.strip() for line in file.readlines()]
        self.stocks = []

    def download_data(self, start_date, end_date=None):
        for ticker in self.tickers:
            print(ticker)
            data = yf.download(ticker, group_by='Ticker', start=start_date, end=end_date)
            data['ticker'] = ticker
            self.stocks.append(data)
            data.to_csv(f'env/data/ticker_{ticker}.csv')

    def read_data(self):
        for ticker in self.tickers:
            data = pd.read_csv(f'env/data/ticker_{ticker}.csv')
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            self.stocks.append(data)

    def load(self, start_date=None, end_date=None):
        if start_date is None:
            self.read_data()
        else:
            self.download_data(start_date, end_date)
        return self.stocks
