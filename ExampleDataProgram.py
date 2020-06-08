import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotmethodsLowVolt as plm
import statsmodels.api as sm 
import scipy as sp


path = r'C:\Users\hcn62\OneDrive\Skrivbord\Exjobb\Exempeldata till exjobb.xlsx'
df = pd.read_excel(path)

L = int(len(df)/3)
df1 = df.loc[0:L, ['RegistrationTime', 'Förbrukning','Hours']] 
df1.reset_index(drop=True, inplace=True)
df1['RegistrationTime'] = pd.to_datetime(df['RegistrationTime'])
df1.set_index('RegistrationTime', inplace=True)
df2 = df.loc[L:L*2, ['RegistrationTime', 'Förbrukning','Hours']] 
df2.reset_index(drop=True, inplace=True)
df2['RegistrationTime'] = pd.to_datetime(df['RegistrationTime'])
df2.set_index('RegistrationTime', inplace=True)
df3 = df.loc[L*2:L*3, ['RegistrationTime', 'Förbrukning','Hours']]
df3.reset_index(drop=True, inplace=True)
df3['RegistrationTime'] = pd.to_datetime(df['RegistrationTime'])
df3.set_index('RegistrationTime', inplace=True)
#plm.findmax(df1)
plm.tempcorr(df2)
#x = np.corrcoef(weather['Lufttemperatur'].values,df1['Förbrukning'].values)
#print(df1)