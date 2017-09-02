import os
import sys
import warnings
import pandas as pd
# import pandas_datareader.data as web
import numpy as np

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from arch import arch_model

import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy.linalg import LinAlgError
from statsmodels import ConvergenceWarning


def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        # mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))

        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return


def best_arima_model(TS, arima_order):
    best_aic = np.inf
    best_order = None
    best_mdl = None

    p_, d_, q_ = arima_order
    p_rng = range(1, p_ + 1)  # [0,1,2,3,4]
    q_rng = range(1, q_ + 1)
    d_rng = range(d_ + 1)  # [0,1]
    for i in p_rng:
        for d in d_rng:
            for j in q_rng:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    try:
                        tmp_mdl = smt.ARIMA(TS, order=(i, d, j)).fit(
                            method='mle', trend='nc', disp=False
                        )
                        tmp_aic = tmp_mdl.aic
                        if tmp_aic < best_aic:
                            best_aic = tmp_aic
                            best_order = (i, d, j)
                            best_mdl = tmp_mdl
                    # except Exception as e:
                    #     continue
                    # except RuntimeWarning as rw:
                    #     continue
                    except (ConvergenceWarning, RuntimeWarning, ValueError, LinAlgError):
                        continue
    # p('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
    return best_aic, best_order, best_mdl


def fit_and_predict(ret, arima_order=(5,0,5), garch_order=(1,1), ahead=1):
    _, best_order, _ = best_arima_model(ret, arima_order)
    ari_p, ari_i, ari_q = best_order
    am = arch_model(ret, p=ari_p, o=ari_i, q=ari_q, dist='StudentsT')
    res = am.fit(update_freq=5, disp='off')
    # print(res.summary())
    pred = res.forecast()
    pred = pred.mean.iloc[-1][0]
    return pred
    # if pred > 0:
    #     return 1
    # else:
    #     return -1

