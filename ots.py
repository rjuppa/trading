import json
import requests
import decimal
import numpy as np
import pandas as pd

import matplotlib
# matplotlib.use('TkAgg')   # use for OSX

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from datetime import datetime, timedelta
import matplotlib.dates as mdates


class OTS(object):
    money = 0.0
    bitcoin = 0.0
    bitcoin_buy_price = 0.0
    bad_buy_limit = 0.90
    commission = 0.9975  # 0.25 %
    period = 50
    do_print = False

    def reset_money(self):
        self.money = 5000  # start money 5000USD
        self.bitcoin = 0.0

    def read_data(self, dt_from, count=2000, from_symbol='BTC', to_symbol='USD', exchange='Poloniex'):
        """ Load dataset from internet """
        ts_now = dt_from.timestamp()
        qs = 'fsym={}&tsym={}&toTs={}&e={}&limit={}'.format(from_symbol, to_symbol,
                                                            int(ts_now), exchange, count)
        if count <= 2000:
            url = 'https://min-api.cryptocompare.com/data/histohour?' + qs
        else:
            url = 'https://min-api.cryptocompare.com/data/histominute?' + qs
        response = requests.get(url)
        data = json.loads(response.content)['Data']
        df = pd.DataFrame(data)
        df.index = df.time.map(lambda x: datetime.fromtimestamp(int(x)))
        df['balance'] = json.dumps({'usd': 0.0, 'btc': 0.0})
        plt.show()
        return df

    def get_presignal(self, row):
        """
        Detects if there is a change in trend
        For a change we need 2 same sequent signals
        """
        if row.color > 1 and row.color_shift > 1 and row.color_shift2 < 1 and row.color_shift3 < 1:
            # 2 : 2 : 0|-2, : 0|-2
            return 1  # turn up
        if row.color < -1 and row.color_shift < -1 and row.color_shift2 > -1 and row.color_shift3 > -1:
            # -2 : -2 : 0|2 : 0|2
            return -1  # turn down
        return 0

    def get_balance(self, row):
        """ Executes a trade upon signal """
        if row.signal == 1:
            self.bitcoin = self.money / row.close
            self.money = 0.0
        if row.signal == -1:
            self.money = self.bitcoin * row.close
            self.bitcoin = 0.0
        return self.money + self.bitcoin * row.close

    def get_signal(self, row):
        """ Processes pre-signals and raise signals """
        # BUY
        if self.money > 0:  # have money
            if row.close < row.buy_line:
                # if price is low under EMA100

                if row.pre_signal == 1:
                    # trend is UP (pre_signal == 1)
                    self.bitcoin = self.money / row.close * self.commission
                    self.bitcoin_buy_price = row.close
                    self.money = 0
                    row.bank = self.money + self.bitcoin * row.close
                    if self.do_print:
                        print('{:%Y-%m-%d %H:%M}  ===  {:+d}   ---   {:8.3f}   ---  {:8.3f}   BUY:({:8.3f})'.format(
                            datetime.fromtimestamp(row.time),
                            int(row.pre_signal), self.money, self.bitcoin, row.close))
                    return 1
        # SELL
        if self.bitcoin > 0:
            if row.close > row.ema100:
                # high price
                if row.pre_signal == -1:
                    # trend is DOWN (pre_signal == -1)
                    self.money = self.bitcoin * row.close * self.commission
                    self.bitcoin = 0
                    if self.do_print:
                        print('{:%Y-%m-%d %H:%M}  ===  {:+d}   ---   {:8.3f}   ---  {:8.3f}   SELL({:8.3f})'.format(
                            datetime.fromtimestamp(row.time), int(row.pre_signal), self.money, self.bitcoin, row.close))
                    return -1
            else:
                # low price
                if row.close < self.bitcoin_buy_price * self.bad_buy_limit:
                    # more then 10% loss - wrong buy
                    self.money = self.bitcoin * row.close * self.commission
                    self.bitcoin = 0.0
                    if self.do_print:
                        print('{:%Y-%m-%d %H:%M}  ===  {:+d}   ---   {:8.3f}   ---  {:8.3f}   SELL({:8.3f})'.format(
                            datetime.fromtimestamp(row.time), int(row.pre_signal), self.money, self.bitcoin, row.close))
                    return -1
        return 0

    def calculate_signal(self, df):
        """
        Calculating all helper data columns
        Mostly based on exponential weighted mean and other rolling functions
        EMA and difference
        """

        df['close_shift'] = df['close'].shift()
        df['close_shift2'] = df['close'].shift(2)

        df['ema100'] = df['close'].ewm(self.period).mean()
        df['average'] = df['ema100'].mean()
        df['ema5'] = df['close'].ewm(5).mean()
        df['ema_buy'] = df['close'].rolling(window=int(self.period / 2), center=True,
                                            min_periods=2).mean() * 0.98
        df['add'] = (df['ema5'] - df['average']) * 0.04     # if there is up hill add more
        df['buy_line'] = (df['ema100'] + df['ema_buy']) / 2 + df['add'].abs()
        df['mean20'] = df['close'].rolling(window=int(self.period / 4), center=True,
                                           min_periods=2).mean()
        df['mean20_shift'] = df['mean20'].shift()

        df['diff2'] = (df['mean20'].diff(periods=1) + df[
            'mean20'] / 100000)  # this is tolerance when derivation is near zero
        df['diff'] = df['diff2'].rolling(window=10, center=True, min_periods=1).mean()

        df['color'] = np.where(df['diff'] < 0, -2, np.where(df['diff'] > 0, 2, 0))
        df['color_shift'] = np.where(df['diff'].shift() < 0, -2,
                                     np.where(df['diff'].shift() > 0, 2, 0))
        df['color_shift2'] = np.where(df['diff'].shift(2) < 0, -2,
                                      np.where(df['diff'].shift(2) > 0, 2, 0))
        df['color_shift3'] = np.where(df['diff'].shift(3) < 0, -2,
                                      np.where(df['diff'].shift(3) > 0, 2, 0))
        df['pre_signal'] = df.apply(self.get_presignal, axis=1)

        if self.do_print:
            print('Starts with:')
            print('Money = 5000 USD')
            print('Coins = 0 BTC')
            print('DateTime          ===  S    ---        USD   ---      Coin    Price in USD')

        df['signal'] = df.apply(self.get_signal, axis=1)
        self.reset_money()
        df['balance'] = df.apply(self.get_balance, axis=1)
        return df

    def draw_graph(self, df, from_symbol='BTC', to_symbol='USD'):
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 8)
        ax.plot(df.index, df['close'], alpha=0.0)

        # convert dates to numbers first
        inxval = mdates.date2num(df.index.to_pydatetime())
        points = np.array([inxval, df['close']]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # colorize line
        cmap = ListedColormap(['r', 'b', 'g'])
        norm = BoundaryNorm([-2, -1, 1, 2], cmap.N)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1)
        lc.set_array(df['color'])
        ax.add_collection(lc)

        # SELL Signal
        below_threshold = df[df['signal'] == -1]
        plt.scatter(below_threshold.index, below_threshold['close'], color='red')

        # BUY Signal
        below_threshold = df[df['signal'] == 1]
        plt.scatter(below_threshold.index, below_threshold['close'], color='darkgreen')

        # helper lines
        # plt.plot(market_data.index, market_data['average'], color='gray')
        # plt.plot(df.index, df['ema100'], color='pink')
        plt.plot(df.index, df['buy_line'], color='orange')

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        monthFmt = mdates.DateFormatter("%b")
        ax.xaxis.set_major_formatter(monthFmt)

        plt.title('{}/{}'.format(from_symbol, to_symbol))
        plt.grid(True)

        xfmt = mdates.DateFormatter('%m-%d %H')
        ax.xaxis.set_major_formatter(xfmt)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))

        h = df['close'].max() - df['close'].min()
        y_min = df['close'].min() - h / 2
        y_max = df['close'].max() + h / 2
        plt.axis([df.index.min() + timedelta(hours=12), df.index.max(), y_min, y_max])
        plt.setp(plt.gca().xaxis.get_majorticklabels(), 'rotation', 90)
        plt.show()

    def find_optimal_bad_buy_limit(self, df):
        bad_buy_limit_data = []
        bad_buy_limit_profit = []
        market_data_copy = df.copy(True)
        for limit in np.arange(0.99, 0.75, -0.01):
            self.bad_buy_limit = limit
            market_data = self.calculate_signal(market_data_copy)
            bad_buy_limit_data.append(self.bad_buy_limit)
            bad_buy_limit_profit.append(int(market_data.tail(1)['balance']))
            self.reset_money()

        bad_buy_limit_df = pd.DataFrame(
            {'bad_buy_limit': bad_buy_limit_data,
             'profit': bad_buy_limit_profit})
        bad_buy_limit_df.plot(x='bad_buy_limit', y='profit')
        plt.title('BAD BUY LIMIT %')
        plt.grid(True)
        return bad_buy_limit_df.ix[bad_buy_limit_df['profit'].idxmax()]['bad_buy_limit']

    def find_optimal_rolling_window(self, df):
        rolling_window_data = []
        rolling_window_profit = []
        market_data_copy = df.copy(True)
        for rolling_window in np.arange(12, 120, 1):
            self.period = rolling_window
            market_data = self.calculate_signal(market_data_copy)
            rolling_window_data.append(self.period)
            rolling_window_profit.append(int(market_data.tail(1)['balance']))
            self.reset_money()

        rolling_window_df = pd.DataFrame(
            {'rolling_window': rolling_window_data,
             'profit': rolling_window_profit})
        rolling_window_df.plot(x='rolling_window', y='profit')
        plt.title('ROLLING WINDOW SIZE')
        plt.grid(True)
        return rolling_window_df.ix[rolling_window_df['profit'].idxmax()]['rolling_window']

    def start_from(self, dt_from, from_symbol='BTC'):
        market_data = self.read_data(dt_from, 2000, from_symbol=from_symbol)
        print('read_data.. done ')

        print('Calculating optimal BAD BUY limit..')
        self.bad_buy_limit = self.find_optimal_bad_buy_limit(market_data)
        print('Optimal bad_buy_limit: {}% of buy price.'.format(self.bad_buy_limit))

        print('Calculating optimal rolling window size..')
        self.period = self.find_optimal_rolling_window(market_data)
        print('Optimal rolling_window: {}'.format(self.period))

        self.do_print = True
        market_data = self.calculate_signal(market_data)
        self.draw_graph(market_data)

if __name__ == "__main__":
    dt_from = datetime.utcnow()
    ots = OTS()
    ots.start_from(dt_from)
