import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotmethodsLowVolt as plm
import station_methods as stm
import statsmodels.api as sm 
import scipy as sp
from sklearn import preprocessing

#### Loads the data from excel files or from h5 file, note the first time it has to be loaded from the excelfiles
dfdict = stm.load_data(load_excel = False)
#plt.show()

XY = stm.create_input(dfdict)
print(XY['nbrCus'].unique())
print(XY.shape)
test = XY.loc[XY['nbrCus'] == 114]
XY = XY.loc[XY['nbrCus'] != 114]
#test = XY['nbrCus'].values[XY['nbrCus'].values == 114]
#XY['nbrCus'] = XY['nbrCus'].values[XY['nbrCus'].values != 114]
print(test.shape)
print(XY.shape)
