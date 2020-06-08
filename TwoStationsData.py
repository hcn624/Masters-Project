import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotmethodsLowVolt as plm
import station_methods as stm
import statsmodels.api as sm 
import scipy as sp
import skextremes as ske

reload = False
if (reload):
    path = r'C:\Users\hcn62\OneDrive\Skrivbord\Exjobb\Data\N316 Januari - November 2018.xlsx'
    path2 = r'C:\Users\hcn62\OneDrive\Skrivbord\Exjobb\Data\N317 Januari - November 2018.xlsx'
    df1 = pd.read_excel(path, na_values = '-')
    df2 = pd.read_excel(path2, na_values = '-')
    df1.to_hdf('dfs.h5', key='df1', mode='w')
    df2.to_hdf('dfs.h5', key='df2')
else:
    df1 = pd.read_hdf('dfs.h5', key= 'df1')
    df2 = pd.read_hdf('dfs.h5', key = 'df2')

df1 = stm.removeOutliers(df1)
df2 = stm.removeOutliers(df2)
#print(df1)
df1['Period'] = pd.to_datetime(df1['Period'])
df2['Period'] = pd.to_datetime(df2['Period'])

#### Plots the dispersion of a station given a date
#fig, ax =stm.plotdispersion(df2,'2018-05-04 00:00:00','2018-05-04 23:00:00')
#print(len(df1['GSRN'].unique()))
#print(len(df2['GSRN'].unique()))
#### Aggregates the customers to create the data for each station
df1 = df1.pivot_table(index = 'Period', values = 'Förbrukning', aggfunc = 'sum')
df2 = df2.pivot_table(index = 'Period', values = 'Förbrukning', aggfunc = 'sum')
# Plots the duration curve of the stations
fig1, ax1 = stm.durCur(df1,1)
fig2, ax2 = stm.durCur(df2,2)
plt.show()
#### Removes non-stationarity
#df1 = stm.removeSeason(df1)
#df2 = stm.removeSeason(df2)
#### Finds block maximas for each station
#maxes = df1.pivot_table(index = pd.Grouper(freq = 'M'), values = 'Förbrukning', aggfunc = 'max')
#maxes2 = df2.pivot_table(index = pd.Grouper(freq = 'M'), values = 'Förbrukning', aggfunc = 'max')
#### Fits a GEV-distribution for each stationarised station
#model = stm.fit_gev(maxes)
#model2 = stm.fit_gev(maxes2)
#plt.show()
###### Checks if the data correlates, it does highly
#stm.correlation(df1, df2)
###### Checks correlation between stations and temperature and fits a linear regression model
#stm.tempcorr(df1)
#stm.tempcorr(df2)



    