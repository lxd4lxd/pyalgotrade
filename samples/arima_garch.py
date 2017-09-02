#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
*******************************************************************************
                              Description

    ARIMA+GARCH MODEL PREDICT STARTEGY.
    1. 测试R接口是否好用
    2. 编写R函数得到最优ARIMA参数
    3. 编写R函数得到GARCH参数
_______________________________________________________________________________
                            Functions List

_______________________________________________________________________________
                        Created on 16:30 2017-08-22

                              @author: xdliu

                            All rights reserved.
*******************************************************************************
"""
#
import os

# import rpy2.robjects as rob
# import rpy2.robjects.numpy2ri
# rpy2.robjects.numpy2ri.activate()
import numpy as np
import pandas as pd
# rob.r.source('arima_garch.R')

def trade_signal(x):
    sig = rob.r.fit_and_predict(x, p=5, i=0, q=5)
    # script = '''
    # #library('rugarch')
    # #library('tseries')
    #
    # # read in csv
    # df = read.csv("%s")
    # # print(str(df))
    # rt = diff(log(df["Close"]))
    # # acf(rt)
    # # print(rt)
    #
    #
    # ''' % csv_path
    # sig = rob.r(script)
    return sig
if __name__ == '__main__':
    path = os.path.abspath('../histdata/commodity/FG0.csv')
    df = pd.read_csv(path)
    cl_rt = np.diff(np.log(df["Close"]))
    window = 200
    for i in range(len(cl_rt) - window):
        rt = cl_rt[i:i+window]
        indicator = trade_signal(rt)
        print('predict: %s' % indicator[0])
        print('    real rt: %s' % cl_rt[i+window])
        # print('real close:')
