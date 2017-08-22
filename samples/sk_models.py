#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
*******************************************************************************
                              Description

    This module extract the single line data in four steps.
    1. Analyze the layout style by axis recognition
    2. Segment the character chain proposal and OCR them
    3. Line detection by color and thin
    4. Data mapping by linear transformation
_______________________________________________________________________________
                            Functions List

    foo: foo has the input of something, output something, used some algorithm.
_______________________________________________________________________________
                        Created on 16:00 2017-07-31

                              @author: xdliu

                            All rights reserved.
*******************************************************************************
"""
import os

import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import pyalgotrade.barfeed.sina_feed as sf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, \
    QuadraticDiscriminantAnalysis
import pandas as pds
import numpy as np


def create_lagged_series(ts, symbol, lags=5):
    """
    This creates a pandas DataFrame that stores the
    percentage returns of the adjusted closing value of
    a stock obtained from Yahoo Finance, along with a
    number of lagged returns from the prior trading days
    (lags defaults to 5 days). Trading volume, as well as
    the Direction from the previous day, are also included.
    """

    # Obtain stock information from Yahoo Finance
    # ts = DataReader(
    #     symbol, "yahoo",
    #     start_date-datetime.timedelta(days=365),
    #     end_date
    # )

    # Create the new lagged DataFrame
    tslag = pds.DataFrame(index=ts.index)
    tslag["Today"] = ts["Close"]
    tslag["Volume"] = ts["Volume"]

    # Create the shifted lag series of prior trading period close values
    for i in range(0,lags):
        tslag["Lag%s" % str(i+1)] = ts["Close"].shift(i+1)

    # Create the returns DataFrame
    tsret = pds.DataFrame(index=tslag.index)
    tsret["Volume"] = tslag["Volume"]
    tsret["Today"] = tslag["Today"].pct_change()*100.0

    # If any of the values of percentage returns equal zero, set them to
    # a small number (stops issues with QDA model in scikit-learn)
    for i,x in enumerate(tsret["Today"]):
        if (abs(x) < 0.0001):
            tsret["Today"][i] = 0.0001

    # Create the lagged percentage returns columns
    for i in range(0,lags):
        tsret[
            "Lag%s" % str(i+1)
            ] = tslag["Lag%s" % str(i+1)].pct_change()*100.0

    # Create the "Direction" column (+1 or -1) indicating an up/down day
    tsret["Direction"] = np.sign(tsret["Today"])
    # tsret = tsret[tsret.index >= start_date]

    return tsret


instrument = 'FG0'
csv_path = os.path.abspath('../histdata/commodity') + '/' + instrument + '.csv'
# feed = sf.Feed()
# feed.addBarsFromCSV(instrument, csv_path)
df = pds.read_csv(csv_path)
l = len(df.index)
window_size = 99
lagged_data = create_lagged_series(df, instrument,
                                   # start_date=datetime.datetime(2012, 12, 03),
                                   # end_date=datetime.datetime(2017, 07, 21),
                                   lags=window_size)
x = lagged_data[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5']]
y = lagged_data['Direction']

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.8
                                                    )
x_train = x_train.fillna(method='ffill')
y_train = y_train.fillna(method='ffill')
x_test = x_test.fillna(method='ffill')
y_test = y_test.fillna(method='ffill')

# lda = LinearDiscriminantAnalysis()
# lda = QuadraticDiscriminantAnalysis()
lda = RandomForestClassifier()
lda.fit(x_train, y_train)

predict = lda.predict(x_test)

hit_rate = lda.score(x_test, y_test)
cm = confusion_matrix(predict, y_test)

print('hit_rate: %0.3f%%' % (hit_rate * 100))
print('confusion matrix:\n%s' %cm)
pass




