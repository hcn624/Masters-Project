import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skextremes as ske
import statsmodels.api as sm 

# Finds the block maximas of df with size of blocks specified by blocksize
def findMaxes(df, blocksize = 'M'):
    df = df.pivot_table(index = 'Period', values = 'Förbrukning', aggfunc = 'sum')
    df = df.pivot_table(index = pd.Grouper(freq = blocksize), values = 'Förbrukning', aggfunc = 'max')
    return df

# Fits the block maxima to the GEV distribution, returning the GEV model
def fit_gev(df):
    #df.plot(marker = 'x')
    L = df.to_numpy().size
    data = df.to_numpy().reshape(L)
    model = ske.models.classic.GEV(data = data, fit_method='mle', ci = 0.05,  ci_method='bootstrap')
    model.plot_summary()
    print(model.stats(moments = 'mvsk'))
    return model

# Removes the seasonal trend and standard deviation in an attempt to reduce non-stationarity
def removeSeason(df, T = 720):
    L = df.to_numpy().size
    std_ar = np.zeros(L)
    av_ar = np.zeros(L)
    for i in range(L):
        av_ar[i], std_ar[i] = helpAv(i, df, L, T)
    df['S_average'] = av_ar
    df['S_std'] = std_ar
    df['Förbrukning'] = (df['Förbrukning'] - df['S_average'])/df['S_std']
    return df

# Calculates the monthly average and standard dev. for each multistep using time window +-T
def helpAv(i, df, L, T):
    if (i - T < 0):
        st = 0
    else:
        st = i - T
    if (i + T >= L):
        en = L-1
    else:
        en = i + T
    av = df['Förbrukning'][st:en+1].mean()
    std = df['Förbrukning'][st:en+1].std()
    return av, std

# Creating a dictionary of dataframes to separate customers
def sepCus(df):
    df_dic = dict()
    st = 0
    prev = df['GSRN'][0]
    for i, e in enumerate(df['GSRN']):
        if (prev != e):
            en = i
            df_dic[e] = df[:][st:en]
            st = en
        prev = e
    return df_dic

#Removes outliers that must be false and inserts instead average of neighbouring values
def removeOutliers(df, threshold = 1000, outlierEqZero = True):
    #df = df[df['Förbrukning'].abs() < threshold]
    df = df.fillna(0)
    # IF outlierEqZero is true turn outsider to zero, else just drop them
    if (outlierEqZero == True):
        df['Förbrukning'].values[df['Förbrukning'].values > threshold] = 0
        df['Förbrukning'].values[df['Förbrukning'].values < -threshold] = 0
    else:
        df = df[df['Förbrukning'] < threshold]
        df = df[df['Förbrukning'] > -threshold]

    return df

#Calculates correlation between two stations
def correlation(df1, df2):
    f = df1['Förbrukning'].to_numpy()
    s = df2['Förbrukning'].to_numpy()
    print(np.corrcoef(f,s))

#Calculates the correlation between dataframe and temperature
def tempcorr(df):
    path = r'C:\Users\hcn62\OneDrive\Skrivbord\Exjobb\Data\smhi2018.xlsx'
    weather = pd.read_excel(path,parse_dates=[['Datum', 'Tid (UTC)']])
    #print(weather)
    weather = pd.merge(left=weather, left_on='Datum_Tid (UTC)',
         right=df, right_on='Period')
    x = np.corrcoef(weather['Lufttemperatur'].values, weather['Förbrukning'].values)
    print(x)
    X = sm.add_constant(weather['Lufttemperatur'].values)
    Y = weather['Förbrukning'].values
    mod = sm.OLS(Y, X)
    res = mod.fit()
    print(res.summary())

#plots all different customers for one day with unique color
def plotdispersion(df, dayst = '2018-05-03 00:00:00', dayen = '2018-05-03 23:00:00'):
    #greater than the start date and smaller than the end date
    mask = (df['Period'] >= dayst) & (df['Period'] <= dayen)
    date = dayst[0:10]
    dfdate = df[:][mask]
    dfdate.reset_index(drop=True, inplace=True)
    print(dfdate)
    size = len(dfdate['GSRN'].unique())
    fig, ax =  plt.subplots()
    for i in range(size):
        st = i*24
        en = (i+1)*24
        data = dfdate['Förbrukning'][st:en]
        data.reset_index(drop=True, inplace=True)
        data.plot(ax = ax, color = 'black')
    ax.set_xticks(range(24))
    ax.set_xlabel(date + ' (Timme)')
    ax.set_ylabel('Förbrukning (kWh)')
    return fig, ax

# Plots the duration curve of a station
def durCur(df, nbr):
    df = df['Förbrukning'].sort_values(axis = 0)
    df.reset_index(drop=True, inplace=True)
    fig, ax = plt.subplots()
    df.plot()
    plt.grid(True)
    plt.title('Varaktighetskurva station ' + str(nbr))
    plt.xlabel('Timvärden sorterade i växande storlek')
    plt.ylabel('Förbrukning (kWh)')
    return fig, ax
# Fits the data with linear regression using nbrCus as explanatory variable
def fitnbrCus(dfdict):
    means = np.zeros((len(dfdict)))
    nbrsCus = np.zeros((len(dfdict)))
    for i ,df in enumerate(dfdict.values()):
        nbrsCus[i] = len(df['GSRN'].unique())
        means[i] = df.pivot_table(index = 'Period', values = 'Förbrukning', aggfunc = 'sum').mean()
    X = sm.add_constant(means)
    mod = sm.OLS(nbrsCus,X)
    res = mod.fit()
    ypred = res.predict(nbrsCus)
    
    plt.plot(means, marker = 'x')
    plt.plot(ypred, marker = 'x')
    print(res.summary())
    return res
# Help function for retreieving the time attributes
def findMonthWeekday(df):
    df['Weekday'] = df.index.weekday
    df['Hour'] = df.index.hour
    df['Month'] = df.index.month
    return df
# Help function for retrieving all weather related attributes
def findWeather(df):
    path = r'C:\Users\hcn62\OneDrive\Skrivbord\Exjobb\Data\smhi2018.xlsx'
    path2 = r'C:\Users\hcn62\OneDrive\Skrivbord\Exjobb\Data\vindhastighet_2018.xlsx'
    path3 = r'C:\Users\hcn62\OneDrive\Skrivbord\Exjobb\Data\luftfuktighet_2018.xlsx'
    weather = pd.read_excel(path,parse_dates=[['Datum', 'Tid (UTC)']])
    #wind = pd.read_excel(path2,parse_dates=[['Datum', 'Tid (UTC)']])
    df = pd.merge(left=df, left_on = 'Period',
        right=weather, right_on='Datum_Tid (UTC)')
    #df = pd.merge(left=df, left_on = 'Datum_Tid (UTC)',
    #    right=wind, right_on='Datum_Tid (UTC)')
    #print(df)
    return df
# Creating the time series input later used for supervised learning
def create_input(dfdict):
    df = pd.DataFrame()
    for sub_df in dfdict.values():
        nbrCus = len(sub_df['GSRN'].unique())
        print(nbrCus)
        sub_agg = sub_df.pivot_table(index = 'Period', values = 'Förbrukning', aggfunc = 'sum')
        sub_agg = findMonthWeekday(sub_agg)
        sub_agg.reset_index()
        sub_agg = findWeather(sub_agg)
        sub_agg.reset_index(drop=True, inplace=True)
        sub_agg['nbrCus'] = nbrCus
        df = df.append(sub_agg)
    df.reset_index(drop=True, inplace=True)
    df = df.drop(columns = 'Datum_Tid (UTC)')
    return df

# Loads the data from excel into a dictionary of pandas dfs, load_excel = True loads the data from excel files and saves them as h5, false loads from h5
# This method also remove all outliers with values of 1000 or under 1000, see function removeOutliers
def load_data(load_excel):
    paths = [r'C:\Users\hcn62\OneDrive\Skrivbord\Exjobb\Data\N316 Januari - November 2018.xlsx', r'C:\Users\hcn62\OneDrive\Skrivbord\Exjobb\Data\N317 Januari - November 2018.xlsx',
        r'C:\Users\hcn62\OneDrive\Skrivbord\Exjobb\Data\N026 Januari - November 2018.xlsx', r'C:\Users\hcn62\OneDrive\Skrivbord\Exjobb\Data\N213 Januari - November 2018.xlsx',
            r'C:\Users\hcn62\OneDrive\Skrivbord\Exjobb\Data\N295 Januari - November 2018.xlsx']
    names = ['df1', 'df2', 'df3', 'df4', 'df5']
    dfdict = {}
    if (load_excel == True):
        for i, (path, name) in enumerate(zip(paths, names)):
            dfdict[name] = pd.read_excel(path, na_values = '-')
            dfdict[name] = removeOutliers(dfdict[name], outlierEqZero = False)
            if (i == 0):
                dfdict[name].to_hdf('dfs.h5', key=name, mode='w')
            else:
                dfdict[name].to_hdf('dfs.h5', key=name)
    else:
        for name in names:
            dfdict[name] = pd.read_hdf('dfs.h5', key=name)
            #plotdispersion(dfdict[name]) ######## <------ NOT NEEDED LATER
    return dfdict


