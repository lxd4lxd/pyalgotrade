#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from datetime import datetime

from pyalgotrade import bar
from pyalgotrade.stratanalyzer import returns
from pyalgotrade.technical import ma
from pyalgotrade.technical import cross
import os

from pyalgotrade import strategy
from pyalgotrade import plotter
from pyalgotrade.barfeed import sina_feed as sf

from pyalgotrade.stratanalyzer import sharpe, drawdown
import numpy as np
import rpy2.robjects as rob
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()
rob.r.source('arima_garch.R')


class ArimaGarch(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument, window=100, arima_order=(5, 0, 5),
                 garch_order=(1, 1)):
        strategy.BacktestingStrategy.__init__(self, feed)
        self.__instrument = instrument
        # self.__longPos = None
        self.__longPos = None
        self.__shortPos = None
        # We'll use adjusted close values instead of regular close values.
        self.setUseAdjustedValues(False)
        self.__prices = feed[instrument].getPriceDataSeries()
        self.__lowDS = feed[instrument]._BarDataSeries__lowDS
        self.__highDS = feed[instrument]._BarDataSeries__highDS
        self.window = window
        self.arima_order = arima_order
        self.garch_order = garch_order
        self.long_exeinfo = True
        self.short_exeinfo = True
        # self.__sma = SMA

    def onEnterCanceled(self, position):
        if self.__longPos == position:
            self.__longPos = None
        elif self.__shortPos == position:
            self.__shortPos = None
        else:
            assert False

    def onExitOk(self, position):
        if self.__longPos == position:
            self.__longPos = None
        elif self.__shortPos == position:
            self.__shortPos = None
        else:
            assert False

    def onExitCanceled(self, position):
        # If the exit was canceled, re-submit it.
        position.exitMarket()

    def on_long_trend(self):
        high1, high2 = self.__highDS[-2:]
        low1, low2 = self.__lowDS[-2:]
        if high2 >= high1 and low2 >= low1:
            return True
        return False

    def on_short_trend(self):
        high1, high2 = self.__highDS[-2:]
        low1, low2 = self.__lowDS[-2:]
        if high2 <= high1 and low2 <= low1:
            return True
        return False

    def onBars(self, bars):

        # self.info(self.__lowDS[-1])
        # self.info(self.__highDS[-1])
        if len(self.__prices) > self.window:

            window_prices = np.array(self.__prices[(-1-self.window):-1])
            rt = np.diff(np.log(window_prices))
            predict = rob.r.fit_and_predict(rt,
                                           p=self.arima_order[0],
                                           i=self.arima_order[1],
                                           q=self.arima_order[2])
            signal = predict[0]
            self.info(signal)
            self.info(self.getBroker().getEquity())
            if self.__longPos is not None:
                if self.long_exeinfo:
                    self.info('long executed on %s' % self.__longPos._Position__entryDateTime)
                    self.long_exeinfo = False
                if not self.on_long_trend() and signal < 0 and \
                        not self.__longPos.exitActive():
                    self.__longPos.exitMarket()
                    self.info('long exited on %s' % bars[self.__instrument].getDateTime())
            elif self.__shortPos is not None:
                if self.short_exeinfo:
                    self.info('short executed on %s' % self.__shortPos._Position__entryDateTime)
                    self.short_exeinfo =False
                if not self.on_short_trend() and signal > 0 and \
                        not self.__shortPos.exitActive():
                    self.__shortPos.exitMarket()
                    self.info('short exited order on %s' % bars[self.__instrument].getDateTime())
            else:
                cash = self.getBroker().getCash()
                self.info(cash)
                shares = int(cash * 0.8 / bars[self.__instrument].getPrice())
                if signal < 0:
                    self.info('short signal on %s' % bars[self.__instrument].getDateTime())
                    self.__shortPos = self.enterShort(self.__instrument, shares, True)
                    self.short_exeinfo = True

                elif signal > 0:
                    self.info('long signal on %s' % bars[self.__instrument].getDateTime())
                    self.__longPos = self.enterLong(self.__instrument, shares, True)
                    self.long_exeinfo = True
            ######################################################
            # if signal > 0:
            #     if self.__shortPos is not None:
            #         self.info(self.__shortPos.getSumitDateTime())
            #         self.__shortPos.exitMarket()
            #         self.info('short signal on: %s' % bars[self.__instrument].getDateTime().strftime('%Y-%m-%d'))
            #         shares = int(self.getBroker().getCash() * 0.8 // bars[self.__instrument].getPrice())
            #     else:
            #         shares = int(self.getBroker().getCash() * 0.4 // bars[self.__instrument].getPrice())
            #     if self.__longPos is None:
            #         cash = self.getBroker().getCash()
            #         self.info(cash)
            #
            #         self.info('long signal on: %s' % bars[self.__instrument].getDateTime().strftime('%Y-%m-%d'))
            #         # Enter a buy market order. The order is good till canceled.
            #         self.__longPos = self.enterLong(self.__instrument, shares, False)
            # else:
            #     if self.__longPos is not None:
            #         self.info(self.__longPos.getSumitDateTime())
            #         self.__longPos.exitMarket()
            #         # self.info('long exit signal on: %s' % bars[self.__instrument].getDateTime().strftime('%Y-%m-%d'))
            #         shares = int(self.getBroker().getCash() * 0.8 // bars[self.__instrument].getPrice())
            #     else:
            #         shares = int(self.getBroker().getCash() * 0.4 // bars[self.__instrument].getPrice())
            #     if self.__shortPos is None:
            #         cash = self.getBroker().getCash()
            #         self.info(cash)
            #         # shares = int(self.getBroker().getCash() * 0.4 / bars[self.__instrument].getPrice())
            #         self.info('short signal on: %s' % bars[self.__instrument].getDateTime().strftime('%Y-%m-%d'))
            #         self.__shortPos = self.enterShort(self.__instrument, shares, False)


def test_strat(plot):
    instrument = 'FG0'
    # plot = True
    # instrument = "600281SH"
    # short_ma_Period = 10
    # medium_ma_period = 20
    # long_ma_period = 99

    # Download the bars.
    # feed = yahoofinance.build_feed([instrument], 2011, 2012, ".")
    csv_path = os.path.abspath('../histdata/commodity') + '/' + instrument + '.csv'
    # feed = sf.Feed(frequency=bar.Frequency.MINUTE)
    feed = sf.Feed()
    feed.addBarsFromCSV(instrument, csv_path)
    # feed = csvfeed.Feed('Date', )
    strat = ArimaGarch(feed, instrument,
                       window=400, arima_order=(5, 0, 5), garch_order=(1, 1))
    sharpeRatioAnalyzer = sharpe.SharpeRatio()
    strat.attachAnalyzer(sharpeRatioAnalyzer)
    returnsAnalyzer = returns.Returns()
    ddAnalyzer = drawdown.DrawDown()
    strat.attachAnalyzer(ddAnalyzer)

    if plot:
        plt = plotter.StrategyPlotter(strat, True, True, True)
        # plt.getInstrumentSubplot("FG0").addDataSeries("close")
        plt.getOrCreateSubplot("returns").addDataSeries("Simple returns", returnsAnalyzer.getReturns())
        # plt.getInstrumentSubplot(instrument).addDataSeries("short sma", strat.getShortSMA())
        # plt.getInstrumentSubplot(instrument).addDataSeries("medium sma", strat.getMediumSMA())
        # plt.getInstrumentSubplot(instrument).addDataSeries("long sma", strat.getLongSMA())

        # plt.getInstrumentSubplot(instrument).addDataSeries("middle", strat.getBollingerBands().getMiddleBand())
        # plt.getInstrumentSubplot(instrument).addDataSeries("lower", strat.getBollingerBands().getLowerBand())

    strat.run()
    print("Sharpe ratio: %.2f" % sharpeRatioAnalyzer.getSharpeRatio(0.05))
    if plot:
        plt.plot()
    pass


if __name__ == "__main__":
    test_strat(True)
