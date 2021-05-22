# -*- coding: utf-8 -*-
"""
Created on Mon May 10 18:38:40 2021
@author: Sahil
"""
#importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture



all_data = pd.read_csv('./files/all_stock_data_with_indicators.csv')
all_data.Date = pd.to_datetime(all_data.Date)
all_data = all_data.set_index('Date')

print(all_data.info())

#Extract the returns
returns = all_data[['symbol','RC']].copy()
returns['Date'] = returns.index.copy()

#Pivot the returns to create series of returns for each stock 
transposed = returns.pivot(index = 'Date', columns = 'symbol', values = 'RC')

#Transpose the data to get companies on the index level and dates on the column level since clusters takes place on index level
X = transposed.dropna().transpose()

#Extract sum of squares for K-means clusters from 1 to 50 clusters
sum_of_sq = np.zeros([30, 1])
for k in range(1, 31):
    sum_of_sq[k-1] = KMeans(n_clusters=k).fit(X).inertia_
    
plt.plot(range(1, 30), sum_of_sq[1:30])
plt.title("Elbow Method") 
plt.xlabel("Number of Cluster") 
plt.ylabel("Within-cluster Sum of Squares")

pd.DataFrame(sum_of_sq, columns = ['Difference in SS'], index = range(1,31)).diff()


#Get 17 clusters
gmm = GaussianMixture(n_components = 15)
gmm.fit(transposed.dropna().transpose())

#Predict for each company
clusters = gmm.predict(transposed.dropna().transpose())
clusters_df = pd.DataFrame({'Cluster':clusters,
                           'Companies':transposed.columns})

#Sort by Clusters
clusters_df = clusters_df.sort_values(['Cluster']).reset_index(drop = True)


#[THIS WILL OVERRIDE ABOVE OPERATION WITH INDUSTRY CLASSIFICATION]
nifty_with_industry = pd.read_csv('./files/ind_nifty100list_with_cnx.csv')
nifty_with_industry['Companies'] = nifty_with_industry['Companies']+'.NS'
clusters_df = pd.read_csv('./files/clusters/clusters.csv')

new_cluster_df = clusters_df.merge(nifty_with_industry,how='left',on=['Companies'])
new_cluster_df['Cluster'] = new_cluster_df['Industry']

clusters_df = new_cluster_df[['Cluster','Companies','NIFTY_INDEX']]

clusters_df = clusters_df.sort_values(['Cluster']).reset_index(drop = True)
#Save as csv
clusters_df.to_csv("./files/clusters/clusters.csv")
