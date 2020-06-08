import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotmethodsLowVolt as plm
import station_methods as stm
import statsmodels.api as sm 
import scipy as sp
from sklearn import preprocessing
from keras import models, layers
from keras.models import load_model
from keras.layers import Dense
import model_methods as modM

#### Loads the data from excel files or from h5 file, note the first time it has to be loaded from the excelfiles
dfdict = stm.load_data(load_excel = True)
#### Removes outliers that are physically impossible by setting them to 0
#df1 = stm.removeOutliers(df1)
#df2 = stm.removeOutliers(df2)
#dfdict = {1 : df1, 2 : df2}
XY = stm.create_input(dfdict)
#model = stm.fitnbrCus(dfdict) 

##### Split into train, vaildation and test set. First randomise the order
XY = XY.sample(frac=1).to_numpy()
train_size = int(len(XY)*0.6)
val_size = int(len(XY)*0.2)
test_size = len(XY) - train_size - val_size
train = XY[0:train_size,:]
val = XY[train_size:train_size+val_size,:]
test = XY[train_size+val_size:,:]


##### Normalise the input and output
trainscaler = preprocessing.StandardScaler().fit(train)
valscaler = preprocessing.StandardScaler().fit(val)
testscaler = preprocessing.StandardScaler().fit(test)
train = trainscaler.transform(train)
val = trainscaler.transform(val)
test = testscaler.transform(test)

# Create the s,y pairs for supervised learning
trainX, trainY = modM.xymaker(train)
valX, valY = modM.xymaker(val)
testX, testY = modM.xymaker(test)

###### Build the model (FFNN)

# trains and saves or loads model depending on hasModel
hasModel = False
if (hasModel == False):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
    input_shape=(trainX.shape[1],)))
    model.add(layers.Dropout(0.2))
    model.add(Dense(12,activation = 'relu'))
    model.add(Dense(1))
    model.compile(loss='mean_absolute_error', optimizer='rmsprop')
    #model.summary()
    history = model.fit(trainX, trainY, validation_data=(valX, valY), epochs = 50)
    model.save('OnlyCustomersTime.h30')
    np.save('testX', testX)
    np.save('testY', testY)
    np.save('valX', valX)
    np.save('valY', valY)
    np.save('trainX', trainX)
    np.save('trainY', trainY)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
else:
    model = load_model('OnlyCustomersTime.h30')
    testX = np.load('testX.npy')
    testY = np.load('testY.npy')
    valX = np.load('valX.npy')
    valY = np.load('valY.npy')
    trainX = np.load('trainX.npy')
    trainY = np.load('trainY.npy')
    
##### Evaluates model and plot residuals and such
modM.evaluation(model, testX, testY, testscaler)
plt.show()
input()


