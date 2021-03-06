# -*- coding: utf-8 -*-
"""
Created on Mon May 17 23:30:53 2021

@author: Sahil
"""
import pandas as pd
import numpy as np
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import datetime as dt
import os

dir = './static/'
for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))
    
    
plt.rcParams['figure.figsize'] = [12, 7]
plt.rc('font', size=14)

all_data = pd.read_csv('./files/all_stock_data_with_indicators.csv')
all_data.Date = pd.to_datetime(all_data.Date)
all_data.Date = all_data.Date.transform(lambda x: x.date())
all_data = all_data.set_index('Date')



all_index_data = pd.read_csv('./files/all_index_data.csv')
all_index_data.Date = pd.to_datetime(all_index_data.Date)
all_index_data.Date = all_index_data.Date.transform(lambda x: x.date())
all_index_data = all_index_data.set_index('Date')



def get_trend_slope(top_10_pred):
   slopes = []
   for index,row in top_10_pred.iterrows(): 
       df = all_data[all_data.symbol == row.Companies]
       selected = df[:row.Date].tail(30)['Close']
       selected = selected.reset_index(drop=True)
       coefficients, residuals, _, _, _ = np.polyfit(range(len(selected.index)),selected,1,full=True)
       mse = residuals[0]/(len(selected.index))
       nrmse = np.sqrt(mse)/(selected.max() - selected.min())
       slopes.append(coefficients[0])
   top_10_pred['slope'] = slopes
   return top_10_pred
       
 

def get_buy_value(top_10_pred):
   buys = []
   for index,row in top_10_pred.iterrows(): 
       df = all_data[all_data.symbol == row.Companies]
       buy_value = df[row.Date:].head(2)['Close'].values[0]
       buys.append(buy_value)
   top_10_pred['buy_value'] = buys
   return top_10_pred


#this method is straight forward; check all the lows of the session days when the trade is ON and anytime the low reached the stoploss update realised returns 
def cut_losses_at(stoploss,top_10_pred):
   realised_pct_change = []
   for index,row in top_10_pred.iterrows(): 
       df = all_data[all_data.symbol == row.Companies]
       selected = df[row.Date:].head(21)['Low'].values[1:]
       if True in np.less_equal(selected,row.buy_value*(1 - stoploss)):
           realised_pct_change.append(-stoploss*100)
       else:
           realised_pct_change.append(row.pct_change_trade)
       
   top_10_pred['pct_change_trade'] = realised_pct_change
   return top_10_pred


#Get the spot support and resistance for the trade
#To get support which is below the buy_value, sort the levels in descending order and get 1st less value of buy_value
#To get the resitance which is above the buy_value, sort the levels in ascending order and get the 1st greater value of the buy_value 
def get_nearest_support_resistance(top_10_pred,plot=False):
    nearest_support = []
    nearest_resistance = []
    risk_reward = []
    for index,row in top_10_pred.iterrows(): 
       levels_df = get_levels(row.Companies, row.Date,plot)
       levels_df = levels_df.sort_values(by=['price'], ascending=False)
       try:
           temp = levels_df[levels_df['price'].lt(row.buy_value)].values[0][0]
           #temp = temp - (0.01*temp)
           nearest_support.append(temp)
       except:
           nearest_support.append('NA')
       levels_df = levels_df.sort_values(by=['price'], ascending=True)
       try:
           temp = levels_df[levels_df['price'].gt(row.buy_value)].values[0][0]
           #temp = temp - (0.01*temp)
           nearest_resistance.append(temp)
       except:
           nearest_resistance.append('NA')
                  
    top_10_pred['nearest_support'] = nearest_support
    top_10_pred['nearest_resistance'] = nearest_resistance
    return top_10_pred


def get_risk_reward(top_10_pred):
    #Configuring to 1:2 risk:reward
    #Get all trades which satisfies above risk:reward ratio
    risk_reward = []
    for index,row in top_10_pred.iterrows():
        if(row.nearest_support == 'NA'):
            ratio = 0;
        else:
            risk = round(row.buy_value - row.nearest_support,2)
            #discard the trade if risk is more than 5%
            risk_pct  = abs(((row.nearest_support-row.buy_value)/row.buy_value)*100)
            if  risk_pct <= 2 :
                ratio = 0
            else:
                try:
                    reward = round(row.nearest_resistance - row.buy_value,2)
                    ratio = round(reward/risk,2)
                except:
                    ratio = 3.0
            
        risk_reward.append(ratio)
    top_10_pred['risk_reward_ratio']  = risk_reward
    return top_10_pred
    
    
    

def get_index_change(top_10_pred):
    date = top_10_pred.Date.values[0]
    data = all_index_data.loc[date]
    index_change = top_10_pred.merge(data[['NIFTY_INDEX','3DRC','1WRC']],on=["NIFTY_INDEX"], how='left')
    index_change['NIFTY50_1WRC'] = data[data['NIFTY_INDEX'] == 'NIFTY 50'].reset_index()['1WRC'].values[0]
    return index_change
        
        
        
        
        
        
#Execute the trade and see if target or stoploss is encounter
    #if the a specific trading day is positive[the diffrence of open and close (close-open) gives positive value] for the particular stock
        #check the high first for the target and then low
    #else check the low first for the stoploss and then high for the target
#assign realised return to the if any one condition matches during the trade otherwise reaised return will be % gain/loss over 30 days
def execute_trade_with_levels(top_10_pred):
   realised_pct_change = []
   exit_log = []
   trade_status = []
   sell_value = []
   for index,row in top_10_pred.iterrows(): 
       df = all_data[all_data.symbol == row.Companies]
       selected = df[row.Date:][['Open','High','Low','Close']]
       selected = selected.iloc[1:]
       
       #Start the game 
       sell = selected.iloc[-1:]['Close'].values[0]
       realised_pct = ((sell-row.buy_value)/row.buy_value)*100
       exit_date = pd.to_datetime(selected.iloc[-1:].index)[0].date()
       status = 'HOLD'
       
       if row.nearest_resistance == 'NA':
           #reward = stoploss x 2; no upper resistance means expecting 3x reward than risk
           reward = abs(((row.nearest_support-row.buy_value)/row.buy_value)*100)*3
           row.nearest_resistance = row.buy_value + ((reward/100)*row.buy_value)
       for i,r in selected.iterrows():
           
           day_sentiment = r['Open'] - r['Close']
           
           #adding trailling stoploss if the target is near with 
           if day_sentiment:
               if r['High'] >= row.nearest_resistance:
                   pct_change_trade = ((row.nearest_resistance-row.buy_value)/row.buy_value)*100
                   realised_pct = pct_change_trade
                   exit_date = i.strftime('%Y-%m-%d')
                   status = 'TARGET'
                   sell = row.nearest_resistance
                   break
               elif r['Low'] <= row.nearest_support:
                   pct_change_trade = ((row.nearest_support-row.buy_value)/row.buy_value)*100
                   realised_pct = pct_change_trade
                   exit_date = i.strftime('%Y-%m-%d')
                   status = 'STOPLOSS'
                   sell = row.nearest_support
                   break
           else:
               if r['Low'] <= row.nearest_support:
                   pct_change_trade = ((row.buy_value-row.nearest_support)/row.nearest_support)*100
                   realised_pct = pct_change_trade
                   exit_date = i.strftime('%Y-%m-%d')
                   status = 'STOPLOSS'
                   sell = row.nearest_support
                   break
               elif r['High'] >= row.nearest_resistance:
                   pct_change_trade = ((row.nearest_resistance-row.buy_value)/row.buy_value)*100
                   realised_pct = pct_change_trade
                   exit_date = i.strftime('%Y-%m-%d')
                   status = 'TARGET'
                   sell = row.nearest_resistance
                   break
       exit_log.append(exit_date)
       trade_status.append(status)
       sell_value.append(sell)
       realised_pct_change.append(realised_pct) 
   top_10_pred['pct_change_trade'] = realised_pct_change
   top_10_pred['exit_date'] = exit_log
   top_10_pred['Status'] = trade_status
   top_10_pred['sell_value'] = sell_value
   return top_10_pred
    
###for all support and resistance from last 120 trading days
def get_levels(ticker,date,plot):
    levels = []
    levels_df = pd.DataFrame(columns =['price'])
    df = all_data[all_data.symbol ==  ticker]
    df = df[:date].tail(180)
    df['Date'] = pd.to_datetime(df.index)
    df['Date'] = df['Date'].apply(mpl_dates.date2num)
    df = df.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]

    s =  np.mean(df['High'] - df['Low'])
    for i in range(2,df.shape[0]-2):
        if isSupport(df,i):
            l = df['Low'][i]
            if isFarFromLevel(l,s,levels):
                levels.append((i,l))
                levels_df = levels_df.append({'price':l},ignore_index=True)
        elif isResistance(df,i):
            l = df['High'][i]
            if isFarFromLevel(l,s,levels):
                levels.append((i,l))
                levels_df = levels_df.append({'price':l},ignore_index=True)
                
    if plot:   
        plot_all(df,levels,ticker)
                
    
    return levels_df
    
    
def isSupport(df,i):
  support = df['Low'][i] < df['Low'][i-1]  and df['Low'][i] < df['Low'][i+1] and df['Low'][i+1] < df['Low'][i+2] and df['Low'][i-1] < df['Low'][i-2]
  return support

def isResistance(df,i):
  resistance = df['High'][i] > df['High'][i-1]  and df['High'][i] > df['High'][i+1] and df['High'][i+1] > df['High'][i+2] and df['High'][i-1] > df['High'][i-2]
  return resistance

def isFarFromLevel(l,s,levels):
   return np.sum([abs(l-x) < s  for x in levels]) == 0

def plot_all(df,levels,ticker):
  levels_list = []
  for i,l in levels:
      levels_list.append(l)
  fig, ax = plt.subplots()
  candlestick_ohlc(ax,df.values,width=0.6, \
                   colorup='green', colordown='red', alpha=0.8)
  date_format = mpl_dates.DateFormatter('%d %b %Y')
  ax.xaxis.set_major_formatter(date_format)
  ax.yaxis.set_ticks(levels_list)
  fig.autofmt_xdate()
  fig.tight_layout()
  plt.title(ticker)
  for level in levels:
    plt.hlines(level[1],xmin=df['Date'][level[0]],xmax=max(df['Date']),colors='blue')
  fig.savefig(f"./static/{ticker}.png")
  #fig.show()
  


