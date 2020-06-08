### Various imports ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm 

""" Plots the duration curves of all customers """
def durCur(df1, df2, df3):
    df1 = df1.sort_values(axis = 0)
    df1.reset_index(drop=True, inplace=True)
    df2 = df2.sort_values(axis = 0)
    df2.reset_index(drop=True, inplace=True)
    df3 = df3.sort_values(axis = 0)
    df3.reset_index(drop=True, inplace=True)
    fig, ax = plt.subplots()
    df1.plot()
    df2.plot()
    df3.plot()
    plt.grid(True)
    plt.legend(['Kund 1', 'Kund 2', 'Kund 3'])
    plt.title('Varaktighetskurvor 3 kunder')
    plt.xlabel('Timvärden sorterade i växande storlek')
    plt.ylabel('Förbrukning (kWh)')

""" Plots the hourly electricity consumption for each customer for the entire year """
def plotCus(df1, df2, df3):
    fig, ax =  plt.subplots(3,1)
    df1.plot(ax=ax[0])
    df2.plot(ax=ax[1])
    df3.plot(ax=ax[2])
    ax[0].legend(['Kund 1'], loc='upper right')
    ax[1].legend(['Kund 2'], loc='upper right')
    ax[1].set(ylabel = 'Förbrukning (kWh)')
    ax[2].legend(['Kund 3'],loc='upper right')
    ax[2].set(xlabel = 'Timvärde')
    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)
    fig.tight_layout()
    fig.suptitle('Årsförbrukning 3 kunder')

""" This help function is used in plotmonthav and returns one plot of weekly data
and the maximum value of that week. """
def plotweek(df, st, en, ax):
    df['Förbrukning'][st*24:en*24].plot(ax= ax, marker = 'x')
    ax.set_xlabel('')
    m = np.max(df['Förbrukning'][st*24:en*24])
    ax.set(ylim = (0, m))
    ax.grid()
    return m, ax

""" This function creates one plot of the data for each customer for each week in the year
and saves the resut as png in a folder, it also plots the weekly maximums for the year. """
def plotmonthav(df1, df2, df3):
    L = math.floor(len(df1)/7/24)
    max = np.zeros((3,47))
    for st in range(L):
        fig, ax =  plt.subplots(3, 1, constrained_layout=True)
        st = st*7
        en = st + 7
        x = (df1, df2 ,df3)
        for i, df in enumerate(x):
            m, ax[i] = plotweek(df, st, en, ax[i])
            max[i][int(st/7)] = m
        ax[1].set_ylabel('Förbrukning (kWh)')
        path = r'C:\Users\hcn62\OneDrive\Skrivbord\Exjobb\Veckoplottar\vecka_' + str(en/7) + '.png'
        fig.savefig(path)
        plt.close("all")
    fig, ax =  plt.subplots(3, 1, constrained_layout=True)
    for i in range(3):
        ax[i].plot(range(1,len(max[i])+1), max[i], marker ='x')
        ax[i].grid()
    path = r'C:\Users\hcn62\OneDrive\Skrivbord\Exjobb\Veckoplottar\veckomax.png'
    ax[1].set_ylabel('Förbrukning (kWh)')
    ax[2].set_xlabel('Vecka')
    fig.suptitle('Maximal förbrukning per vecka')

    fig.savefig(path)
    plt.close("all")

""" This function plots the frequency information of 1 customer
in lin scale and log scale using the Welch method, note that it uses
1 whole year of data even though the data might not be stationary """
def freqinfo(df, customer):
    summer = df['Förbrukning']
    #summer = df[:][3624:5832]
    summer -= summer.mean()
    pxx, freqs = plt.psd(summer.squeeze().values, scale_by_freq = False, noverlap = 128, detrend = 'linear')
    plt.clf()
    plt.close('all')
    fig, ax =  plt.subplots(1,2,constrained_layout=True)
    freqs = np.array(freqs)/2
    ax[0].plot(freqs,pxx)
    ax[0].grid()
    ax[0].set_xlabel('Freq (1/hour)')
    ax[0].set_ylabel('Lin power')
    ax[1].plot(freqs,np.log(pxx))
    ax[1].grid()
    ax[1].set_xlabel('Freq (1/hour)')
    ax[1].set_ylabel('Log power')
    fig.suptitle('Frequency content Welch 50' + chr(37) + ' overlap Customer ' +str(customer))
    return (fig, ax)

""" This function plots a qq-plot for all customers showing if the load of each hour, eg the 3th,
on the day is normally distributed or not in summer. """
def plothourqqnorm(df1, df2, df3):
    x = (df1, df2, df3)
    for h in range(24):
        fig, ax =  plt.subplots(3, 1, constrained_layout=True)
        for nbr, df in enumerate(x):
            df = df.assign(timme = np.mod(range(len(df)),24))
            summer = df[['Förbrukning','timme']][3624:5832]
            summer = summer.sort_values(by=['timme']).drop('timme', axis = 1).squeeze().values
            L = len(summer)
            summer = summer.reshape(24,int(L/24))
            tre = summer[h]
            tre = (tre - np.mean(tre))/np.std(tre)
            sm.qqplot(tre, line ='45', ax = ax[nbr])
        path = r'C:\Users\hcn62\OneDrive\Skrivbord\Exjobb\QQplottar\qqplot_timme_'+ str(h) + '.png'
        fig.suptitle('QQ plot summer hour ' + str(h))
        fig.savefig(path)
        plt.close("all")

def findmax(df,option = None):
    if option == 'week':
        L = math.floor(len(df)/7/24)
        vec = np.zeros((L,2))
        for w in range(L):
            i = df['Förbrukning'][w*24*7:(w+1)*24*7].idxmax()
            vec[w][0] = df['Förbrukning'][i]
            vec[w][1] = df['Hours'][i].hour
        dfm = pd.DataFrame(data = vec, columns=["Förbrukning max", "Tid max"])
        path = r'C:\Users\hcn62\OneDrive\Skrivbord\Exjobb\maxweek.csv'
        dfm.to_csv(path)
        print(dfm['Tid max'].describe())
    else:
        L = math.floor(len(df)/24)
        vec = np.zeros((L,2))
        for d in range(L):
            i = df['Förbrukning'][d*24:(d+1)*24].idxmax()
            vec[d][0] = df['Förbrukning'][i]
            vec[d][1] = df['Hours'][i].hour
        dfm = pd.DataFrame(data = vec, columns=["Förbrukning max", "Tid max"])
        path = r'C:\Users\hcn62\OneDrive\Skrivbord\Exjobb\maxday.csv'
        dfm.to_csv(path)
        print(dfm.describe())

def tempcorr(df):
    path = r'C:\Users\hcn62\OneDrive\Skrivbord\Exjobb\smhi-opendata_1_52350_20200512_140203.xlsx'
    weather = pd.read_excel(path,parse_dates=[['Datum', 'Tid (UTC)']])
    #print(weather)
    weather = pd.merge(left=weather, left_on='Datum_Tid (UTC)',
         right=df, right_on='RegistrationTime')
    x = np.corrcoef(weather['Lufttemperatur'].values, weather['Förbrukning'].values)
    print(x)
    X = sm.add_constant(weather['Lufttemperatur'].values)
    Y = weather['Förbrukning'].values
    mod = sm.OLS(Y, X)
    res = mod.fit()
    print(res.summary())

