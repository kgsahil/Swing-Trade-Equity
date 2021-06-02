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
from flask import Flask ,request, render_template
from markupsafe import Markup
import json

app = Flask(__name__)


clusters_df = pd.read_csv('./files/clusters/clusters.csv')
clusters_df = clusters_df[['Cluster','Companies','NIFTY_INDEX']]



Target_variables = ['SMA_ratio','ATR_5','ATR_15','ATR_Ratio',
                       'ADX_5','ADX_15','SMA_Volume_Ratio','Stochastic_5','Stochastic_15','Stochastic_Ratio',
                      'RSI_5','RSI_15','RSI_ratio','MACD']



# -----------------------------------------------PREDICT----------------------------------


@app.route("/recommend")
def recommend():
    date = request.args['date']
    rr = float(request.args['rr'])
    result  =  get_recommendations(date,rr,True)
    table_template ="""<table class="table table-centered table-nowrap mb-0"><thead class="thead-light">
   <tr>
      <th>Company Name</th>
      <th>Entry</th>
      <th>Target</th>
      <th>Stop Loss</th>
      <th>RR Ratio</th>
       <th>Call</th>
      <th>Details</th>
   </tr>
</thead>"""

    if len(result) == 0:
        table_template += "</table>"
        return table_template
    else:
        table_template += "<tbody>"
        for index,row in result.iterrows():
            Target = 0
            Stoploss = 0
            Entry = str(round(row.buy_value,2))
            if type(row.nearest_resistance) is np.float64 or type(row.nearest_resistance) is float:
                Target = str(round(row.nearest_resistance,2))
            else:
                Target = str(row.nearest_resistance)
            
            if type(row.nearest_support) is np.float64 or type(row.nearest_support) is float:
                Stoploss = str(round(row.nearest_support,2))
            else:
                Stoploss = str(row.nearest_support)
            
            table_template += f"""<tr>
              <td class="text-body font-weight-bold">{row.Companies}</td>
              <td>{Entry}</td>
              <td>{Target}</td>
              <td>{Stoploss}</td>
              <td>1:{row.risk_reward_ratio}</td>
              <td class="font-weight-bold" style="color:#34c38f;">Buy</td>
              <td>
                     <!-- Button trigger modal -->
                     <button type="button" onClick="javascript:window.open('../static/{row.Companies}.png', '_blank');" class="btn btn-primary btn-sm btn-rounded waves-effect waves-light">
                     View Details
                     </button>
                  </td>
               </tr>"""
             
       
    table_template += "</tbody></table>"
    
    data =Markup(table_template)
    
    return render_template('index.html',data=data)
       
    #result  =  get_recommendations(date,rr,False).to_json(orient="records")
    #parsed = json.loads(result)
    #return json.dumps(parsed)  




@app.route("/recommendseq")
def recommendseq():
    
    
    start_date = request.args['startdate']
    end_date = request.args['enddate']
    rr = float(request.args['rr'])
    result  =  get_recommendations_seq(start_date,end_date,rr,True)
    table_template ="""<table class="table table-centered table-nowrap mb-0"><thead class="thead-light">
   <tr>
      <th>Date</th>
      <th>Company Name</th>
      <th>Entry</th>
      <th>Target</th>
      <th>Stop Loss</th>
      <th>RR Ratio</th>
       <th>Call</th>
      <th>Details</th>
   </tr>
</thead>"""
    if len(result) == 0:
        table_template += "</table>"
        return table_template
    else:
        table_template += "<tbody>"
        for index,row in result.iterrows():
            Target = 0
            Stoploss = 0
            Entry = str(round(row.buy_value,2))
            if type(row.nearest_resistance) is np.float64 or type(row.nearest_resistance) is float:
                Target = str(round(row.nearest_resistance,2))
            else:
                Target = str(row.nearest_resistance)
            
            if type(row.nearest_support) is np.float64 or type(row.nearest_support) is float:
                Stoploss = str(round(row.nearest_support,2))
            else:
                Stoploss = str(row.nearest_support)
            
            table_template += f"""<tr>
              <td>{row.Date}</td>
              <td class="text-body font-weight-bold">{row.Companies}</td>
              <td>{Entry}</td>
              <td>{Target}</td>
              <td>{Stoploss}</td>
              <td>1:{row.risk_reward_ratio}</td>
              <td class="font-weight-bold" style="color:#34c38f;">Buy</td>
              <td>
                     <!-- Button trigger modal -->
                     <button type="button" onClick="javascript:window.open('../static/{row.Companies}.png', '_blank');" class="btn btn-primary btn-sm btn-rounded waves-effect waves-light">
                     View Details
                     </button>
                  </td>
               </tr>"""
       
    table_template += "</tbody></table>"
    return table_template





def get_recommendations(Trade_Date,risk_reward_ratio,admin): 
    Trade_Date = dt.datetime.strptime(Trade_Date,'%Y-%m-%d').date()
    try:
        day_data = all_data.loc[Trade_Date]
    except:
        return pd.DataFrame()
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
    top_10_pred = get_buy_value(top_10_pred.copy()) 
    top_10_pred = get_nearest_support_resistance(top_10_pred.copy(),False)
    top_10_pred = get_risk_reward(top_10_pred.copy())
    
    top_10_pred = get_index_change(top_10_pred.copy())
    try: 
         
        #filtering 
        top_10_pred = top_10_pred[top_10_pred.risk_reward_ratio >= risk_reward_ratio] 
        top_10_pred = top_10_pred[top_10_pred['1WRC'] >= top_10_pred['NIFTY50_1WRC']]
       
        if len(top_10_pred) == 0:
              raise Exception("Sorry, no trades")
        if admin:
            _ = get_nearest_support_resistance(top_10_pred.copy(),True)
        return top_10_pred
    except:
        return pd.DataFrame()
    
    
def get_recommendations_seq(start_date,end_date,risk_reward_ratio,admin):
        
    # isoweekday: Monday is 1 and Sunday is 7
    trading_holidays= ['04-03-2019','21-03-2019','17-04-2019','19-04-2019','20-04-2019','01-05-2019','05-06-2019','12-08-2019','15-08-2019','02-09-2019','10-09-2019','02-10-2019','08-10-2019','21-10-2019','28-10-2019','12-11-2019','25-12-2019','21-02-2020','10-03-2020','02-Apr-2020','06-Apr-2020','10-04-2020','14-04-2020','01-05-2020','25-05-2020','02-10-2020','16-11-2020','30-11-2020','25-12-2020','26-01-2021','11-03-2021','20-03-2021','02-04-2021','14-04-2021','21-04-2021','13-05-2021','21-07-2021','19-08-2021','10-09-2021','15-10-2021','04-11-2021','05-11-2021','19-11-2021']
    start_date = dt.datetime.strptime(start_date,'%Y-%m-%d')
    end_date = dt.datetime.strptime(end_date,'%Y-%m-%d')
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
    #print("Trading Days = {}".format(sorted(valid_date_list)))
    number_of_trades = int(len(valid_date_list))
    print('Number of possible trading days : ',number_of_trades)
    
    
    
    stockCounter =  pd.DataFrame()
    TradeBook = pd.DataFrame()
    
    for i in range(0,int(len(valid_date_list))):
        Trade_Date = dt.datetime.strptime(valid_date_list[i], '%d-%m-%Y').date()
        Trade_Date = dt.datetime.strptime(dt.datetime.strftime(Trade_Date, '%Y-%m-%d'),'%Y-%m-%d').date()
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
        
        top_10_pred = get_buy_value(top_10_pred.copy()) 
        top_10_pred = get_nearest_support_resistance(top_10_pred.copy(),False)
        top_10_pred = get_risk_reward(top_10_pred.copy())
        top_10_pred = get_index_change(top_10_pred.copy())  
        
        top_10_pred = top_10_pred[top_10_pred.risk_reward_ratio >= risk_reward_ratio] 
        top_10_pred = top_10_pred[top_10_pred['1WRC'] >= top_10_pred['NIFTY50_1WRC']]
      
        try:
      
          
          if len(top_10_pred) == 0:
              raise Exception("Sorry, no trades")
          print(top_10_pred)
          TradeBook = TradeBook.append(top_10_pred,ignore_index=True)
          if admin:
            _ = get_nearest_support_resistance(top_10_pred.copy(),True)
        except:
          continue
      
    return TradeBook
     
        
if __name__ == "__main__":
    app.run()
    #print(get_recommendations('2021-05-05',1,True))
    #TradeBook =  get_recommendations_seq('2021-05-01','2021-05-10',1,False)

