# -*- coding: utf-8 -*-
"""
Created on Mon May 10 18:44:01 2021

@author: Sahil
"""
#importing packages
import pandas as pd
import numpy as np
import datetime as dt
import os   
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mstats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  TimeSeriesSplit,ShuffleSplit , validation_curve
from sklearn.tree import export_graphviz
from sklearn import tree
import pickle
from sklearn.model_selection import GridSearchCV
sns.set()

from hidden_markov_model import get_volatility


clusters_df = pd.read_csv('./files/clusters/clusters.csv')
clusters_df = clusters_df[['Cluster','Companies']]

all_data = pd.read_csv('./files/all_stock_data_with_indicators.csv')
all_data.Date = pd.to_datetime(all_data.Date)
all_data = all_data.set_index('Date')

all_data['Close_Shifted'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.shift(-19))
all_data['Target'] = ((all_data['Close_Shifted'] - all_data['Open'])/(all_data['Open']) * 100).shift(-1)
all_data['Target_Direction'] = np.where(all_data['Target']>0,1,0)
all_data = all_data.dropna().copy()

Target_variables = ['SMA_ratio','ATR_5','ATR_15','ATR_Ratio',
                       'ADX_5','ADX_15','SMA_Volume_Ratio','Stochastic_5','Stochastic_15','Stochastic_Ratio',
                      'RSI_5','RSI_15','RSI_ratio','MACD']


for variable in Target_variables:
    all_data.loc[:,variable] = mstats.winsorize(all_data.loc[:,variable], limits = [0.1,0.1])


train_data = all_data.loc[:'2020-03-31',]
test_data = all_data.loc['2020-04-01':]

#-----------------------------------------CROSS VALIDATION FOR BETTER HYPERPARAMETER---------------------
#Separate between X and Y
X_train = train_data.loc[:,Target_variables]
Y_train = train_data.loc[:,['Target_Direction']]

rf = RandomForestClassifier()
train_scoreNum, test_scoreNum = validation_curve(rf,
                                X = X_train.loc['2010-01-01':], 
                                y = Y_train.loc['2010-01-01':,'Target_Direction'], 
                                param_name = 'n_estimators', 
                                param_range = [70,100,125,150], cv = ShuffleSplit(n_splits = 3))

train_scores_mean = np.mean(train_scoreNum, axis=1)
train_scores_std = np.std(train_scoreNum, axis=1)
test_scores_mean = np.mean(test_scoreNum, axis=1)
test_scores_std = np.std(test_scoreNum, axis=1)

plt.figure(figsize = (20,10))
plt.plot([70,100,125,150],train_scores_mean)
plt.plot([70,100,125,150],test_scores_mean)
plt.legend(['Train Score','Test Score'], fontsize = 'large')
plt.title('Validation Curve Score for n_estimators', fontsize = 'large')



##------------------------------------------TRAINING----------------------------------------


model_accuracy = pd.DataFrame(columns = ['Cluster','Stocks','Accuracy_score'])

for cluster_selected in clusters_df.Cluster.unique():
    
    print(f'The current cluster training is : {cluster_selected}')
    
    #Get data for that cluster
    co_data = all_data[all_data.symbol.isin(clusters_df.loc[clusters_df.Cluster==cluster_selected,'Companies'].tolist())].copy()
    co_train = co_data.loc[:'2020-03-31']
    co_train = co_train.dropna().copy()
    co_train = co_train.sample(frac=1).reset_index(drop=True)
    

    
    X_train = co_train.loc[:,Target_variables]
    Y_train = co_train.loc[:,['Target_Direction']]

    #Define paramters from Validation Curve
    params = {
          'max_features': ['auto','sqrt'],
          'min_samples_leaf': [10, 15],
          'n_estimators': [100,125,175],
         'min_samples_split':[20, 25, 30]} #Using Validation Curves

    rf = RandomForestClassifier()

    #Perform a TimeSeriesSplit on the dataset
    time_series_split = TimeSeriesSplit(n_splits = 3)
    shuffle_split = ShuffleSplit(n_splits = 3)

    
    rf_cv = GridSearchCV(rf, params, cv = shuffle_split, n_jobs = -1, verbose = 20,refit=True)

    #Fit the random forest with our X_train and Y_train
    rf_cv.fit(X_train, Y_train)
    #print(f'For cluster {cluster_selected} which has {len(clusters_df[clusters_df.Cluster == cluster_selected])} stocks')
    print(f'Best params: {rf_cv.best_params_}')
    #print(f'Best score: {rf_cv.best_score_}')
    #feat_importances = pd.Series(rf_cv.best_estimator_.feature_importances_, index=X_train.columns)
    #feat_importances.plot(kind='barh',title=f'Cluster_{cluster_selected}')
    model_accuracy.loc[len(model_accuracy)]  = [cluster_selected,len(clusters_df[clusters_df.Cluster == cluster_selected]),rf_cv.best_score_]

    #Save the fited variable into a Pickle file
    file_loc = f'./files/clusters/Cluster_{cluster_selected}' 
    pickle.dump(rf_cv, open(file_loc,'wb'))
    #fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (8,8), dpi=800)
    #tree.plot_tree(rf_cv.best_estimator_.estimators_[0],
    #            feature_names=X_train.columns,
    #           filled=True,
    #            rounded=True)
    #fig.savefig(f'Cluster_{cluster_selected}.png')
    
  
print(model_accuracy)

model_accuracy.to_csv('./files/clusters/model_accuracy.csv')