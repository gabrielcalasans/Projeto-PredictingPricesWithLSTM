import pandas as pd
import plotly.express as px
from copy import copy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import yfinance as yfin

class StockData:
    def __init__(self, name, startDate, endDate, target = 1, trainPoint = 0.65):
        self.name = name
        self.startDate = startDate
        self.endDate= endDate
        self.baseStockInfo = self.downloadData()
        self.stockInfoTarget = self.setTarget(target)
        self.scaledData = self.scaleData()
        self.point = trainPoint
        self.trainTest = {}
        self.trainTest = self.splitData()
        
        
        
    def downloadData(self):
        stockData = yfin.download(self.name, start = self.startDate, end = self.endDate)
        return stockData
    
    def setTarget(self, n = 1):
        '''Set n days as target to predict'''
        stockInfoTarget = self.baseStockInfo
        stockInfoTarget = stockInfoTarget[['Adj Close', 'Volume']]
        stockInfoTarget = stockInfoTarget.rename(columns = {
            'Adj Close': 'Close'
        })
        stockInfoTarget['Target'] = stockInfoTarget[['Close']].shift(-n)
        return stockInfoTarget[:-n]

    def scaleData(self):
        scaler = MinMaxScaler(feature_range = (0, 1))
        dfScaled = scaler.fit_transform(self.stockInfoTarget)
        return dfScaled
    
    
    def splitData(self):
        x = []
        y = []
        for i in range(1, len(self.scaledData)):
            x.append(self.scaledData[i-1:i, 0])
            y.append(self.scaledData[i, 0])
        x = np.asarray(x)
        y = np.asarray(y)
        split = int(self.point * len(x))
        xTrain = x[:split]
        yTrain = y[:split]
        xTest = x[split:]
        yTest = y[split:]
        return {
            'xTrain': xTrain, 
            'yTrain': yTrain,
            'xTest': xTest,
            'yTest': yTest,
            'x': x,
            'y': y
        }

    
    
class PredictLSTM:
    def __init__(self, stockData, epochs = 20, units = 150, dropOut = 0.3, batchSize = 32, valSplit = 0.2):
        self.stockData = stockData
        self.epochs = epochs
        self.units = units
        self.dropOut = dropOut
        self.stockData.trainTest['xTrain'], self.stockData.trainTest['xTest'] = self.reshapeArrays()
        self.modelo = self.criaModelo()
        self.modelo.fit(self.stockData.trainTest['xTrain'], self.stockData.trainTest['yTrain'], epochs = epochs, batch_size = batchSize, validation_split = valSplit)
        self.predicted = self.predictions() 
        
    
    
    def reshapeArrays(self, dim = 1):
        novosVetores = []
        vetores = [self.stockData.trainTest['xTrain'], self.stockData.trainTest['xTest']]
        for vetor in vetores:
            vetor = np.reshape(vetor, (vetor.shape[0], vetor.shape[1], dim))
            novosVetores.append(vetor)
        return novosVetores
    
    
    def criaModelo(self):
        inputs = keras.layers.Input(shape = (self.stockData.trainTest['xTrain'].shape[1], self.stockData.trainTest['xTrain'].shape[2]))
        x = keras.layers.LSTM(self.units, return_sequences = True)(inputs)
        x = keras.layers.Dropout(self.dropOut)(x)
        x = keras.layers.LSTM(self.units, return_sequences = True)(x)
        x = keras.layers.Dropout(self.dropOut)(x)
        x = keras.layers.LSTM(self.units)(x)
        output = keras.layers.Dense(1, activation = 'linear')(x)
        model = keras.Model(inputs = inputs, outputs = output)
        model.compile(optimizer = 'adam', loss = 'mse')
        return model
    
    def predictions(self):
        predicoes = self.modelo.predict(self.stockData.trainTest['x'])
        close, testPredicted = self.ajustaDados(predicoes)
        dfPredicted = pd.DataFrame()
        dfPredicted['Date'] = self.stockData.stockInfoTarget.index[1:]
        dfPredicted['Predictions'] = testPredicted
        dfPredicted['Close'] = close[1:]
        return dfPredicted
    
    def ajustaDados(self, predictions):
        testPredicted = []
        for i in predictions:
            testPredicted.append(i[0])

        close = []
        for i in self.stockData.scaledData:
            close.append(i[0])

        return close, testPredicted
    
    
    def interactivePlot(self):
        df = self.predicted
        title = f"{self.stockData.name} - Original vs Predictions"
        fig = px.line(title = title)
        for i in df.columns[1:]:
            fig.add_scatter(x = df['Date'], y = df[i], name = i)

        return fig
        # fig.show()