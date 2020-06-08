from sklearn import preprocessing
from keras import models, layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Creates the X and Y for supervised learning
def xymaker(XY):
    #print(XY.shape)
    X = XY[:,1:]
    Y = XY[:,0]
    #print(X.shape)
    #print(Y.shape)
    return X, Y

# Evaluates the model using the test data and inverse_transforms the results using scaler
# Also plots the resiudals in unordered, ordered mode plus absolute residuals and statistics
def evaluation(model, testX, testY, scaler):
    test_loss = model.evaluate(testX, testY)
    print(test_loss)
    test_loss = test_loss*scaler.scale_[0]
    print(test_loss)
    pred = model.predict(testX).reshape(-1)
    
    res = testY - pred
    print(res.shape)
    print(res)
    res = res*scaler.scale_[0]
    sorted = np.sort(res)
    plt.close('all')
    plt.figure(1)
    plt.plot(res)
    plt.xlabel('Sample')
    plt.ylabel('Residual (kWh)')
    plt.title('Residuals Unordered')
    plt.figure(2)
    plt.plot(sorted)
    plt.xlabel('Sample ordered in acending order')
    plt.ylabel('Residual (kWh)')
    plt.title('Residuals Ordered')
    res = pd.Series(sorted)
    print(res.describe())
    res = res.abs().sort_values()
    res.reset_index(drop=True, inplace=True)
    print(res)
    print(res.describe())
    plt.figure(3)
    res.plot()
    plt.grid(True)
    plt.title('Absolute value of Residuals')
    plt.xlabel('Sample ordered in acending order')
    plt.ylabel('Absolute value of Residual (kWh)')
    pred = pred * scaler.scale_[0] + scaler.mean_[0]
    pred = pd.Series(pred)
    plt.figure(4)
    pred.plot()
    plt.title('Predicted values')
    plt.xlabel('Hours')
    plt.ylabel('Load (kWh)')
    