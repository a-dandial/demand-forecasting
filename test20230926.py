# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 10:58:43 2023

@author: arshdeepd
"""

import pandas as pd
import numpy as np
%matplotlib inline

from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

def wmape(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()

constant = 10

data = pd.read_csv("C:/Users/ArshdeepD/usagetest.csv")

melt = data.melt(id_vars='Part Number', var_name='Date', value_name='Usage')

melt['Date'] = pd.to_datetime(melt['Date'])

melt.head()

melt = melt.rename(columns={'Part Number':'unique_id', 'Date':'ds', 'Usage':'y'})

train = melt

'''
train = melt.loc[melt['ds']< '2023-01-09']

valid = melt.loc[(melt['ds'] >= '2023-01-09') & (melt['ds'] < '2023-08-28')]

valid = valid.to_csv('c:/Users/ArshdeepD/usagetest2.csv', encoding='utf-8')

h = valid['ds'].nunique()
'''
train['y'] += constant

sf = StatsForecast(df = train, models = [AutoARIMA(season_length=52)] , freq = 'W')

sf.fit(train)

#p = sf.forecast(h=80)

p = sf.predict(h = 85, level=[90])

p[['AutoARIMA']] -= constant

p = p.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')

p.head()

p[['AutoARIMA]] = p[['AutoARIMA]]

#wmape_ = wmape(melt['y'].values, p['AutoARIMA'].values)
#print(f'WMAPE: {wmape_:.2%}')

p = p.to_csv('c:/Users/ArshdeepD/forecast2.csv', encoding='utf-8')
