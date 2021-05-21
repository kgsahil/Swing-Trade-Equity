# -*- coding: utf-8 -*-
"""
Created on Sun May 16 12:18:09 2021

@author: Sahil
"""


#importing packages
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mstats
import pickle
sns.set()

#from hidden_markov_model import get_volatility
from helpers import get_buy_value,get_trend_slope,cut_losses_at, get_levels, get_nearest_support_resistance, get_risk_reward


pd.options.display.max_columns = None
pd.options.display.max_rows = None

clusters_df = pd.read_csv('./files/clusters/clusters.csv')
clusters_df = clusters_df[['Cluster','Companies']]

all_data = pd.read_csv('./files/all_stock_data_with_indicators.csv')
all_data.Date = pd.to_datetime(all_data.Date)
all_data = all_data.set_index('Date')

all_data['Close_Shifted'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.shift(-30))
all_data['Target'] = ((all_data['Close_Shifted'] - all_data['Open'])/(all_data['Open']) * 100).shift(-1)
all_data['Target_Direction'] = np.where(all_data['Target']>0,1,0)
all_data = all_data.dropna().copy() 

Target_variables = ['SMA_ratio','ATR_5','ATR_15','ATR_Ratio',
                       'ADX_5','ADX_15','SMA_Volume_Ratio','Stochastic_5','Stochastic_15','Stochastic_Ratio',
                      'RSI_5','RSI_15','RSI_ratio','MACD']

for variable in Target_variables:
    all_data.loc[:,variable] = mstats.winsorize(all_data.loc[:,variable], limits = [0.1,0.1])


train_data = all_data.loc[:'2018-12-31',]
test_data = all_data.loc['2019-01-01':]


# to test individual helper functions 
test_df = pd.DataFrame()
test_df_trend = get_trend_slope(test_df)      
test_df_buy_value = get_buy_value(test_df)
test_df_stoploss = cut_losses_at(0.05,test_df_buy_value)
test_list_levels = get_levels('ACC.NS','2019-01-01')

test_df_get_nearest_support_resistance = get_nearest_support_resistance(test_df)

test_df_risk_reward = get_risk_reward(test_df_get_nearest_support_resistance)

#testing -----------------------------------------------TESTING----------------------------------


# isoweekday: Monday is 1 and Sunday is 7
trading_holidays= ['04-03-2019','21-03-2019','17-04-2019','19-04-2019','20-04-2019','01-05-2019','05-06-2019','12-08-2019','15-08-2019','02-09-2019','10-09-2019','02-10-2019','08-10-2019','21-10-2019','28-10-2019','12-11-2019','25-12-2019','21-02-2020','10-03-2020','02-Apr-2020','06-Apr-2020','10-04-2020','14-04-2020','01-05-2020','25-05-2020','02-10-2020','16-11-2020','30-11-2020','25-12-2020','26-01-2021','11-03-2021','20-03-2021','02-04-2021','14-04-2021','21-04-2021','13-05-2021','21-07-2021','19-08-2021','10-09-2021','15-10-2021','04-11-2021','05-11-2021','19-11-2021']
start_date = dt.date(2020, 10, 5)
end_date = dt.date(2020, 10, 20)
days = end_date - start_date
valid_date_list = {(start_date + dt.timedelta(days=x)).strftime('%d-%m-%Y')
                        for x in range(days.days+1)
                        if (start_date + dt.timedelta(days=x)).isoweekday() <= 5}

valid_date_list = list(valid_date_list)
for d in trading_holidays:
  try:
    valid_date_list.remove(d)
  except:
    continue


valid_date_list.sort(key = lambda date: dt.datetime.strptime(date, '%d-%m-%Y'))
print("Trading Days = {}".format(sorted(valid_date_list)))
number_of_trades = int(len(valid_date_list))
print('Number of possible trades: ',number_of_trades)


pct_day_change = []
investment_returns = []
stockCounter =  pd.DataFrame()
total_investment = 0
for i in range(0,int(len(valid_date_list))):
    Trade_Date = dt.datetime.strptime(valid_date_list[i], '%d-%m-%Y').strftime('%Y-%m-%d')
    day_data = test_data.loc[Trade_Date]

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
    #top_10_pred = pred_for_tomorrow[pred_for_tomorrow['prediction'] >= 0.65].copy()
    top_10_pred.reset_index(drop=True,inplace = True)

   
    
    for selected_company in top_10_pred['Companies']:
        actual = all_data[all_data.symbol == selected_company].loc[Trade_Date,'Target_Direction']
        pct_change = all_data[all_data.symbol == selected_company].loc[Trade_Date,'Target']
        #ADX_15 = all_data[all_data.symbol == selected_company].loc[Trade_Date,'ADX_15']
        #ADX_5 = all_data[all_data.symbol == selected_company].loc[Trade_Date,'ADX_5']
        top_10_pred.loc[top_10_pred['Companies'] == selected_company,'actual'] = actual
        top_10_pred.loc[top_10_pred['Companies']  == selected_company,'pct_change_trade'] = pct_change
        #top_10_pred.loc[top_10_pred['Companies']  == selected_company,'ADX_15'] = ADX_15
        #top_10_pred.loc[top_10_pred['Companies']  == selected_company,'ADX_15'] = ADX_15
        test_df = top_10_pred      
    
    
    
    try:
        
      
      #top_10_pred = cut_losses_at(0.1, top_10_pred)
      top_10_pred = get_buy_value(top_10_pred)
      top_10_pred = get_nearest_support_resistance(top_10_pred)
      top_10_pred = get_risk_reward(top_10_pred)
      top_10_pred = top_10_pred[top_10_pred.risk_reward_ratio >= 1.5]
      print(top_10_pred[['Date','Companies','prediction','buy_value','pct_change_trade','risk_reward_ratio']])
      if len(top_10_pred) == 0:
          raise Exception("Sorry, no trades")
      
      pct_day_change.append(top_10_pred['pct_change_trade'].sum())
      investment_returns.append(top_10_pred['buy_value'].sum()+((top_10_pred['pct_change_trade'].sum()/100)*top_10_pred['buy_value'].sum()))
      total_investment += top_10_pred['buy_value'].sum()
    except:
      number_of_trades -= 1
      continue

plt.plot(pct_day_change)
plt.ylabel('Returns')
plt.xlabel(f'Trades Net profit/loss: {sum(pct_day_change)/number_of_trades}')
plt.title(f'Trades from {start_date} to {end_date}')
print('Number of Trades: ',number_of_trades)
print("Net Profit/Loss: ",sum(pct_day_change)/number_of_trades)
#total_investment = number_of_trades*fix_investment
total_returns_including_principal = sum(investment_returns)
print(total_investment,total_returns_including_principal)
  

