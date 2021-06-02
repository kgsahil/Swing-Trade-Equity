# -*- coding: utf-8 -*-
"""
Created on Wed May 26 22:28:22 2021

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
test_prev_data = pd.DataFrame()


df_prev_index_data = pd.read_csv('./files/all_index_data.csv')
df_prev_index_data = df_prev_index_data.set_index('Date')


for st in indices:
    
    start_date  = dt.datetime.strptime(df_prev_index_data[df_prev_index_data.NIFTY_INDEX == st].tail(1).index[0],'%d-%m-%Y').date()+ dt.timedelta(days=1)
    test_prev_data = df_prev_index_data[df_prev_index_data.NIFTY_INDEX == st].tail(7)
    
    if start_date != dt.date.today():
        
        df_index_data = get_history(symbol=st,index=True,start=start_date,end= dt.date.today())
        df_index_data['NIFTY_INDEX'] = st
    
        df_index_data['Date'] = df_index_data.index
        
        df_index_data.Date = df_index_data.Date.transform(lambda x: x.strftime('%d-%m-%Y'))
        df_index_data = df_index_data.set_index('Date')
        test_prev_data = test_prev_data.append(df_index_data)
        df_all_index_data = df_all_index_data.append(test_prev_data)


try:
    
    df_all_index_data['3DRC'] = df_all_index_data.groupby('NIFTY_INDEX')['Close'].transform(lambda x: x.pct_change(periods = 3)*100) 
    df_all_index_data['1WRC'] = df_all_index_data.groupby('NIFTY_INDEX')['Close'].transform(lambda x: x.pct_change(periods = 7)*100) 
    df_all_index_data =  df_all_index_data.dropna()
    
    df_prev_index_data = df_prev_index_data.append(df_all_index_data)
    
    df_prev_index_data.to_csv("./files/all_index_data.csv")
except:
    print('Data is already up-to-date')

#df_all_index_data['Date'] = pd.to_datetime(df_all_index_data['Date'])
#df_all_index_data['Date']= df_all_index_data['Date'].apply(lambda d: d.strftime('%Y-%m-%d'))
    
    
df_prev_index_data.tail()