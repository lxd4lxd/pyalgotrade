# --*-- coding=utf-8 --*--
# read commodity hist data from sina js interface

import urllib
import json
import os
import csv
from pyalgotrade.barfeed import csvfeed
from pyalgotrade import bar
from pyalgotrade.barfeed import common
from pyalgotrade.utils import dt
import datetime

SINA_API_URL = 'http://stock2.finance.sina.com.cn/futures/api/json.php' \
               '/IndexService.getInnerFuturesDailyKLine?symbol='
DATA_DIR = os.path.abspath('../../histdata/commodity')
instrument = 'FG0'


def comm_daily_data(instrument):
    fg_url = SINA_API_URL + instrument
    commodity_his_data = urllib.urlopen(fg_url)

    # 通过urllib模块中的urlopen的方法打开url
    commodity_his_data = commodity_his_data.read()
    commodity_his_data = json.loads(commodity_his_data)

    csv_path = DATA_DIR + '/' + instrument + '.csv'
    header = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    count = 0
    f = open(csv_path, 'w')
    csv_writer = csv.writer(f)

    for d in commodity_his_data:
        if count == 0:
            csv_writer.writerow(header)
            count += 1
        csv_writer.writerow(d)
    f.close()


def parse_date(date, frequency=bar.Frequency.DAY):
    # Sample: 2005-12-30
    # This custom parsing works faster than:
    # datetime.datetime.strptime(date, "%Y-%m-%d")

    year = int(date[0:4])
    month = int(date[5:7])
    day = int(date[8:10])
    if frequency == bar.Frequency.MINUTE:
        hour = int(date[11:13])
        minute = int(date[14:16])
        second = int(date[17:19])
        ret = datetime.datetime(year, month, day, hour, minute, second)
    else:
        ret = datetime.datetime(year, month, day)
    return ret


class RowParser(csvfeed.RowParser):
    def __init__(self, dailyBarTime, frequency, timezone=None, sanitize=False, barClass=bar.BasicBar):
        self.__dailyBarTime = dailyBarTime
        self.__frequency = frequency
        self.__timezone = timezone
        self.__sanitize = sanitize
        self.__barClass = barClass

    def __parseDate(self, dateString):
        ret = parse_date(dateString, self.__frequency)
        # Time on Yahoo! Finance CSV files is empty. If told to set one, do it.
        # if self.__dailyBarTime is not None:
        #     ret = datetime.datetime.combine(ret, self.__dailyBarTime)
        # Localize the datetime if a timezone was given.
        if self.__timezone:
            ret = dt.localize(ret, self.__timezone)
        return ret

    def getFieldNames(self):
        # It is expected for the first row to have the field names.
        return None

    def getDelimiter(self):
        return ","

    def parseBar(self, csvRowDict):
        dateTime = self.__parseDate(csvRowDict["Date"])
        close = float(csvRowDict["Close"])
        open_ = float(csvRowDict["Open"])
        high = float(csvRowDict["High"])
        low = float(csvRowDict["Low"])
        volume = float(csvRowDict["Volume"])

        if self.__sanitize:
            open_, high, low, close = common.sanitize_ohlc(open_, high, low, close)

        return self.__barClass(dateTime, open_, high, low, close, volume, adjClose=False, frequency=self.__frequency)


class Feed(csvfeed.BarFeed):
    """A :class:`pyalgotrade.barfeed.csvfeed.BarFeed` that loads bars from CSV files downloaded from Yahoo! Finance.

    :param frequency: The frequency of the bars. Only **pyalgotrade.bar.Frequency.DAY** or **pyalgotrade.bar.Frequency.WEEK**
        are supported.
    :param timezone: The default timezone to use to localize bars. Check :mod:`pyalgotrade.marketsession`.
    :type timezone: A pytz timezone.
    :param maxLen: The maximum number of values that the :class:`pyalgotrade.dataseries.bards.BarDataSeries` will hold.
        Once a bounded length is full, when new items are added, a corresponding number of items are discarded from the
        opposite end. If None then dataseries.DEFAULT_MAX_LEN is used.
    :type maxLen: int.

    .. note::
        Yahoo! Finance csv files lack timezone information.
        When working with multiple instruments:

            * If all the instruments loaded are in the same timezone, then the timezone parameter may not be specified.
            * If any of the instruments loaded are in different timezones, then the timezone parameter must be set.
    """

    def __init__(self, frequency=bar.Frequency.DAY, timezone=None, maxLen=None):
        if isinstance(timezone, int):
            raise Exception("timezone as an int parameter is not supported anymore. Please use a pytz timezone instead.")

        if frequency not in [bar.Frequency.MINUTE, bar.Frequency.DAY, bar.Frequency.WEEK]:
            raise Exception("Invalid frequency.")

        super(Feed, self).__init__(frequency, maxLen)

        self.__timezone = timezone
        self.__sanitizeBars = False
        self.__barClass = bar.BasicBar
        # if frequency == bar.Frequency.MINUTE:
        #     self.__dailyTime = datetime.time(0,0,0,0,0)

    def setBarClass(self, barClass):
        self.__barClass = barClass

    def sanitizeBars(self, sanitize):
        self.__sanitizeBars = sanitize

    def barsHaveAdjClose(self):
        return True

    def addBarsFromCSV(self, instrument, path, timezone=None):
        """Loads bars for a given instrument from a CSV formatted file.
        The instrument gets registered in the bar feed.

        :param instrument: Instrument identifier.
        :type instrument: string.
        :param path: The path to the CSV file.
        :type path: string.
        :param timezone: The timezone to use to localize bars. Check :mod:`pyalgotrade.marketsession`.
        :type timezone: A pytz timezone.
        """

        if isinstance(timezone, int):
            raise Exception("timezone as an int parameter is not supported anymore. Please use a pytz timezone instead.")

        if timezone is None:
            timezone = self.__timezone

        rowParser = RowParser(
            self.getDailyBarTime(), self.getFrequency(), timezone, self.__sanitizeBars, self.__barClass
        )
        super(Feed, self).addBarsFromCSV(instrument, path, rowParser)


if __name__ == '__main__':
    csv_path = DATA_DIR + '/' + instrument + '.csv'
    feed = Feed()
    feed.addBarsFromCSV(instrument, csv_path)
    pass