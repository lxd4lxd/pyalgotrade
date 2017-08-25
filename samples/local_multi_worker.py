#!/usr/bin/env python3
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
                        Created on 20:54 2017-08-25

                              @author: xdliu

                            All rights reserved.
*******************************************************************************
"""
from pyalgotrade.optimizer import local
from pyalgotrade.barfeed import sina_feed as sf
import itertools
import os
from samples.statstical import ArimaGarch


def param_generator():
    instrument = ["FG0"]
    window = range(250, 888)
    return itertools.product(instrument, window)


if __name__ == '__main__':
    instrument = "FG0"
    feed = sf.Feed()
    csv_path = os.path.abspath('../histdata/commodity') + '/' + instrument + '.csv'
    # feed = sf.Feed(frequency=bar.Frequency.MINUTE)
    feed = sf.Feed()
    feed.addBarsFromCSV(instrument, csv_path)
    local.run(ArimaGarch, feed, param_generator())
