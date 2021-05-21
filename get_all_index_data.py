# -*- coding: utf-8 -*-
"""
Created on Mon May 10 18:51:01 2021

@author: Sahil
"""

import pandas as pd
import datetime as dt
from nsepy import get_history

indices = ['NIFTY 50','NIFTY BANK', 'NIFTY INFRA', 
 'NIFTY REALTY', 'NIFTY ENERGY', 'NIFTY FMCG', 'NIFTY PHARMA', 'NIFTY PSU BANK', 
 'NIFTY SERV SECTOR', 'NIFTY IT', 'NIFTY AUTO', 'NIFTY MEDIA', 'NIFTY METAL', 'NIFTY COMMODITIES', 'NIFTY CONSUMPTION', 'NIFTY FIN SERVICE']

df_index_data = pd.DataFrame()
df_all_index_data = pd.DataFrame()
for st in indices:
    df_index_data = get_history(symbol=st,index=True,start=dt.date(2018,1,1),end=dt.date(2021,5,7))
    df_index_data['symbol'] = st
    df_all_index_data = df_all_index_data.append(df_index_data)

    
df_all_index_data.head()
df_all_index_data['1DRC'] = df_all_index_data.groupby('symbol')['Close'].transform(lambda x: x.pct_change(periods = 1)*100) 
df_all_index_data['1WRC'] = df_all_index_data.groupby('symbol')['Close'].transform(lambda x: x.pct_change(periods = 7)*100) 
df_all_index_data['1MRC'] = df_all_index_data.groupby('symbol')['Close'].transform(lambda x: x.pct_change(periods = 30)*100) 
df_all_index_data =  df_all_index_data.dropna()
df_all_index_data.index = pd.to_datetime(df_all_index_data.index)
df_all_index_data.to_csv("./files/all_index_data.csv")
