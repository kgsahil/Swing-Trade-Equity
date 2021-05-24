# -*- coding: utf-8 -*-
"""
Created on Mon May 24 16:29:47 2021

@author: Sahil
"""

#importing packages
import pandas as pd
import numpy as np
import datetime as dt
import pickle
from helpers import *

pd.options.display.max_columns = None
pd.options.display.max_rows = None

clusters_df = pd.read_csv('./files/clusters/clusters.csv')
clusters_df = clusters_df[['Cluster','Companies','NIFTY_INDEX']]

all_data = pd.read_csv('./files/all_stock_data_with_indicators.csv')
all_data.Date = pd.to_datetime(all_data.Date)
all_data = all_data.set_index('Date')


Target_variables = ['SMA_ratio','ATR_5','ATR_15','ATR_Ratio',
                       'ADX_5','ADX_15','SMA_Volume_Ratio','Stochastic_5','Stochastic_15','Stochastic_Ratio',
                      'RSI_5','RSI_15','RSI_ratio','MACD']



# -----------------------------------------------PREDICT----------------------------------

#inputs from the user

risk_reward_ratio = 1.5
day_predict_day = dt.date(2020,10,20)
Trade_Date = day_predict_day.strftime('%Y-%m-%d')
day_data = all_data.loc[Trade_Date]

pred_for_tomorrow = pd.DataFrame({'Date':[],
                              'Companies':[],
                              'prediction':[]})

#Predict each stock using the 2nd January Data
for cluster_selected in clusters_df.Cluster.unique():
    rf_cv =  pickle.load(open(f'./files/clusters/Cluster_{cluster_selected}', 'rb'))
    best_rf = rf_cv.best_estimator_
    cluster_data = day_data.loc[day_data.symbol.isin(clusters_df.loc[clusters_df.Cluster==cluster_selected,'Companies'].tolist())].copy()
    cluster_data = cluster_data.dropna()
    if (cluster_data.shape[0]>0):
        X_test = cluster_data.loc[:,Target_variables]

        pred_for_tomorrow = pred_for_tomorrow.append(pd.DataFrame({'Date':cluster_data.index,
                                                                   'Companies':cluster_data['symbol'],
                                                                   'prediction':best_rf.predict_proba(X_test)[:,1]}), ignore_index = True)
top_10_pred = pred_for_tomorrow.sort_values(by = ['prediction'], ascending = False).head(10)
top_10_pred.reset_index(drop=True,inplace = True)
top_10_pred['NIFTY_INDEX'] = top_10_pred.merge(clusters_df[['NIFTY_INDEX','Companies']],how='left', on='Companies')['NIFTY_INDEX']

try: 
    top_10_pred = get_buy_value(top_10_pred.copy()) 
    top_10_pred = get_nearest_support_resistance(top_10_pred.copy(),False)
    top_10_pred = get_risk_reward(top_10_pred.copy())
    top_10_pred = get_index_change(top_10_pred.copy())  
    
    
    #filtering 
    top_10_pred = top_10_pred[top_10_pred.risk_reward_ratio >= risk_reward_ratio] 
    top_10_pred = top_10_pred[top_10_pred['1WRC'] >= top_10_pred['NIFTY50_1WRC']]
    
    if len(top_10_pred) == 0:
          raise Exception("Sorry, no trades")
    _ = get_nearest_support_resistance(top_10_pred.copy(),True)
    print(top_10_pred)
except:
    print('no trades')
  

