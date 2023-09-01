# Databricks notebook source
import os
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, DateType
import random
import scipy.stats as stats
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet
from xgboost import XGBRegressor
from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from datetime import date,datetime
import holidays
today = datetime.now()
today_dt = today.strftime("%d-%m-%Y")

import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

from pmdarima import auto_arima

# COMMAND ----------

years = [2018,2019,2021,2022,2023,2024]
festivos = []

for year in years:
    colombia_holidays = holidays.Colombia(years=year)
    festivos += [x.strftime("%Y-%m-%d") for x in colombia_holidays.keys()]

# COMMAND ----------

dwDatabase = dbutils.secrets.get(scope='gascaribe', key='dwh-name')
dwServer = dbutils.secrets.get(scope='gascaribe', key='dwh-host')
dwUser = dbutils.secrets.get(scope='gascaribe', key='dwh-user')
dwPass = dbutils.secrets.get(scope='gascaribe', key='dwh-pass')
dwJdbcPort = dbutils.secrets.get(scope='gascaribe', key='dwh-port')
dwJdbcExtraOptions = ""
sqlDwUrl = "jdbc:sqlserver://" + dwServer + ".database.windows.net:" + dwJdbcPort + ";database=" + dwDatabase + ";user=" + dwUser + ";password=" + dwPass + ";" + dwJdbcExtraOptions
storage_account_name = dbutils.secrets.get(scope='gascaribe', key='bs-name')
blob_container = dbutils.secrets.get(scope='gascaribe', key='bs-container')
blob_storage = storage_account_name + ".blob.core.windows.net"
config_key = "fs.azure.account.key."+storage_account_name+".blob.core.windows.net"
blob_access_key = dbutils.secrets.get(scope='gascaribe', key='bs-access-key')
spark.conf.set(config_key, blob_access_key)

# COMMAND ----------

query = 'SELECT * FROM ComercializacionML.DatosEDA'

# COMMAND ----------

df = spark.read \
  .format("com.databricks.spark.sqldw") \
  .option("url", sqlDwUrl) \
  .option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
  .option("forwardSparkAzureStorageCredentials", "true") \
  .option("maxStrLength", "1024" ) \
  .option("query", query) \
  .load()

rawData = df.toPandas()


# COMMAND ----------

def process_inputs(df):
    df = df.copy()
    
    ## Filtrar por tipo de usuario
    #df = df[df['TipoUsuario'] == tipoUsuario]
    
    festivosBin = []
    for fecha in df['Fecha']:
        if str(fecha) in festivos:
            festivosBin.append(1)
        else:
            festivosBin.append(0)
            
    df['Festivos'] = festivosBin
    
    
    # Cambiar fecha por datetime
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    
    # Reemplazar valores que salen como 0E-8 a 0.
    df['VolumenM3'] = df['VolumenM3'].replace({0E-8:0})
    
    
    uniqueDispositivos = df['IdDispositivo'].unique()
    
    newDisp = pd.DataFrame(columns=['IdDispositivo', 'Nombre', 'TipoUsuario', 'Red', 'Fecha', 'VolumenM3','VolumenCumSum'])
    
    for disp in uniqueDispositivos:
        dfDispSum = df[df['IdDispositivo'] == disp].sort_values(by='Fecha')
        try:
            dfDispSum['VolumenCumSum'] = dfDispSum['VolumenM3'].cumsum().astype('float')
        except:
            print(disp)
        
        newDisp = pd.concat([dfDispSum,newDisp],axis=0)
    
    newDisp['VolumenM3'] = newDisp['VolumenM3'].astype('float')
    newDisp['Festivos'] = newDisp['Festivos'].astype('int')
    

    return newDisp
    

# COMMAND ----------

X = process_inputs(rawData)
X.shape

# COMMAND ----------

def predict_estaciones(df):

    estacionScore = {}
    for estacion in df['Nombre'].unique():
        print(f'PREDICCION PARA {estacion}')
        prophetdf = df[df['Nombre'] == estacion].reset_index(drop=True).sort_values(by='Fecha')[['Fecha','VolumenM3']]


        prophetdf.columns = ['ds','y']

        prophetTrain = prophetdf[prophetdf['ds'] < '2023-04-01']
        prophetTest = prophetdf[prophetdf['ds'] >= '2023-04-01']


        try:
            model = Prophet()
            model.fit(prophetTrain)

            future_dates = model.make_future_dataframe(periods=prophetTest.shape[0], freq='d')

            forecast = model.predict(future_dates)

            newForecast = forecast[forecast['ds'] >= '2023-04-01'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

            newForecast['yreal'] = prophetTest['y']
            newForecast['error'] = abs(newForecast['yreal'] - newForecast['yhat'])
            newForecast['yhat'] = newForecast['yhat'].apply(lambda x: 0 if x < 0 else x)
        
            error = newForecast['error'].sum()/len(newForecast)
            errorDelDiaSiguiente = float(newForecast.head(1)['error'].values[0])
            errorAbsolutoAlDiaSiguiente = abs((newForecast.head(1)['yreal'].values[0] - newForecast.head(1)['yhat'].values[0])/newForecast.head(1)['yhat'].values[0])

            print(f'El MAE: {error} m3')
            print(f'El diferencia del dia siguiente es: {errorDelDiaSiguiente} m3')
            print(f'El error absoluto al dia siguiente es: {errorAbsolutoAlDiaSiguiente:.2f}%')

            estacionScore[estacion] = ['Prophet Forecasting',error,errorDelDiaSiguiente,errorAbsolutoAlDiaSiguiente]
        except:
            print(f'XXXXXXXXXXXXX ERROR PRODUCIDO POR {estacion} XXXXXXXXXXXXXXXXXXXXXXX')



        # Plot forecasts
        #plt.figure(figsize=(15,8))

        #plt.plot(newForecast['yhat'],label='Predicted')
        #plt.plot(newForecast['yreal'],label=f'True value ({estacion})')
        #plt.legend()

        #plt.show()

    return estacionScore

# COMMAND ----------

estacionScore = predict_estaciones(X)

# COMMAND ----------

estacionScore

# COMMAND ----------

df = pd.DataFrame.from_dict(estacionScore, columns=['Modelo','MAE','Diferencia del Primer Dia','Error Absoluto del Primer Dia'],orient='index')
df['Error Absoluto del Primer Dia'] = df['Error Absoluto del Primer Dia'].apply(lambda x: x*100)
df['Error Absoluto del Primer Dia'] = df['Error Absoluto del Primer Dia'].apply(lambda x: float("{:.2f}".format(x)))
df['MAE'] = df['MAE'].apply(lambda x: float("{:.2f}".format(x)))
df['Diferencia del Primer Dia'] = df['Diferencia del Primer Dia'].apply(lambda x: float("{:.2f}".format(x)))

# COMMAND ----------

df[df['Error Absoluto del Primer Dia'] <= 10]

# COMMAND ----------

# MAGIC %md
# MAGIC ### **ARIMA**

# COMMAND ----------

ARIMAfit = auto_arima(X[X['Nombre'] == 'PONEDERA']['VolumenM3'],trace=True,suppress_warnings=True)

# COMMAND ----------

def predict_estaciones_ARIMA(df):

    estacionScore = {}
    for estacion in df['Nombre'].unique():
        print(f'PREDICCION PARA {estacion}')
        prophetdf = df[df['Nombre'] == estacion].reset_index(drop=True).sort_values(by='Fecha')[['Fecha','VolumenM3']]



        prophetdf.columns = ['ds','y']

        

        start = len(prophetTrain)
        end = len(prophetTrain) +len (prophetTest)-1

        try:
            model = ARIMA(prophetTrain['y'],order=(5,1,5))
            model = model.fit()

            forecast = model.predict(start=start,end=end,typ='levels')

            errorLoop = []
            for i,fc in enumerate(forecast.values):
                errorLoop.append(abs(prophetTest['y'].values[i] - fc))
        
            error = sum(errorLoop)/len(forecast)
            errorDelDiaSiguiente = float(forecast.head(1).values[0])
            errorAbsolutoAlDiaSiguiente = abs((prophetTest.head(1)['y'].values[0] - forecast.head(1).values[0])/forecast.head(1).values[0])

            print(f'El MAE: {error} m3')
            print(f'El diferencia del dia siguiente es: {errorDelDiaSiguiente} m3')
            print(f'El error absoluto al dia siguiente es: {errorAbsolutoAlDiaSiguiente:.2f}%')

            estacionScore[estacion] = ['ARIMA',error,errorDelDiaSiguiente,errorAbsolutoAlDiaSiguiente]
        except Exception as e: 
            print(e)
            print(f'XXXXXXXXXXXXX ERROR PRODUCIDO POR {estacion} XXXXXXXXXXXXXXXXXXXXXXX')



        # Plot forecasts
        #plt.figure(figsize=(15,8))

        #plt.plot(newForecast['yhat'],label='Predicted')
        #plt.plot(newForecast['yreal'],label=f'True value ({estacion})')
        #plt.legend()

        #plt.show()

    return estacionScore

# COMMAND ----------

estacionScoreARIMA = predict_estaciones_ARIMA(X)

# COMMAND ----------

dfARIMA = pd.DataFrame.from_dict(estacionScoreARIMA, columns=['Modelo','MAE','Diferencia del Primer Dia','Error Absoluto del Primer Dia'],orient='index')
dfARIMA['Error Absoluto del Primer Dia'] = dfARIMA['Error Absoluto del Primer Dia'].apply(lambda x: x*100)
dfARIMA['Error Absoluto del Primer Dia'] = dfARIMA['Error Absoluto del Primer Dia'].apply(lambda x: float("{:.2f}".format(x)))
dfARIMA['MAE'] = dfARIMA['MAE'].apply(lambda x: float("{:.2f}".format(x)))
dfARIMA['Diferencia del Primer Dia'] = dfARIMA['Diferencia del Primer Dia'].apply(lambda x: float("{:.2f}".format(x)))

# COMMAND ----------

dfARIMA[dfARIMA['Error Absoluto del Primer Dia'] <= 10]

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Redes Neuronales**

# COMMAND ----------

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)

class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x




prophetdf = X[X['Nombre'] == 'CANNON ESTACION'].reset_index(drop=True).sort_values(by='Fecha')[['Fecha','VolumenM3']]


prophetdf.columns = ['ds','y']

prophetTrain = prophetdf[prophetdf['ds'] < '2023-04-01'][['y']].values.astype('float32')
prophetTest = prophetdf[prophetdf['ds'] >= '2023-04-01'][['y']].values.astype('float32')

lookback = 1
X_train, y_train = create_dataset(prophetTrain, lookback=lookback)
X_test, y_test = create_dataset(prophetTest, lookback=lookback)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

model = AirModel()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)
n_epochs = 2000
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 100 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))


# COMMAND ----------

len(y_pred)

# COMMAND ----------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
import tensorflow as tf

# COMMAND ----------

prophetdf = X[X['Nombre'] == 'CANNON ESTACION'].reset_index(drop=True).sort_values(by='Fecha')[['Fecha','VolumenM3']]


prophetdf.columns = ['ds','y']

prophetTrain = prophetdf[prophetdf['ds'] < '2023-04-01'][['y']].values.astype('float32')
prophetTest = prophetdf[prophetdf['ds'] >= '2023-04-01'][['y']].values.astype('float32')


prophetTrain = np.reshape(prophetTrain,(len(prophetTrain),1,1))

model = Sequential()
model.add(LSTM(
    512,
    input_shape=(prophetTrain.shape[1], prophetTrain.shape[2]),
    return_sequences=False
))
model.add(Dense(512))
#model.add(Dense(numPitches))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='mse', optimizer='rmsprop', metrics=[tf.keras.metrics.RootMeanSquaredError()])

# COMMAND ----------

num_epochs = 10


history = model.fit(np.arange(len(prophetTrain)), prophetTrain, epochs=num_epochs, batch_size=32)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **XGBoost**

# COMMAND ----------

def predict_estacionesXGBoost(df):

    estacionScore = {}
    for estacion in df['Nombre'].unique():
        print(f'PREDICCION PARA {estacion}')
        prophetdf = df[df['Nombre'] == estacion].reset_index(drop=True).sort_values(by='Fecha')[['VolumenM3']]

        prophetdf = prophetdf.replace({'[REDACTED].0':np.nan})

        prophetdf = prophetdf.dropna()

        prophetdf.columns = ['y']
        prophetdf['target'] = prophetdf['y'].shift(-1)

        trainLen = int(len(prophetdf)*0.9)

        prophetTrain = prophetdf.head(trainLen).dropna()
        prophetTest = prophetdf.tail((len(prophetdf) - trainLen)).dropna()

        X = prophetTrain['y']
        y = prophetTrain['target']


        try:
            model = XGBRegressor(objective='reg:squarederror',n_estimators=2000)
            model.fit(X,y)

            predictions = model.predict(prophetTest['y'])

            prophetTest['yhat'] = predictions

            prophetTest['error'] = abs(prophetTest['target'] - prophetTest['yhat'])
            prophetTest['yhat'] = prophetTest['yhat'].apply(lambda x: 0 if x < 0 else x)
        
            error = prophetTest['error'].sum()/len(prophetTest)
            errorDelDiaSiguiente = float(prophetTest.head(1)['error'].values[0])
            errorAbsolutoAlDiaSiguiente = abs((prophetTest.head(1)['target'].values[0] - prophetTest.head(1)['yhat'].values[0])/prophetTest.head(1)['yhat'].values[0])

            print(f'El MAE: {error} m3')
            print(f'El diferencia del dia siguiente es: {errorDelDiaSiguiente} m3')
            print(f'El error absoluto al dia siguiente es: {errorAbsolutoAlDiaSiguiente:.2f}%')

            estacionScore[estacion] = ['XGBoost',error,errorDelDiaSiguiente,errorAbsolutoAlDiaSiguiente]
        except:
            print(f'XXXXXXXXXXXXX ERROR PRODUCIDO POR {estacion} XXXXXXXXXXXXXXXXXXXXXXX')



        # Plot forecasts
        #plt.figure(figsize=(15,8))

        #plt.plot(newForecast['yhat'],label='Predicted')
        #plt.plot(newForecast['yreal'],label=f'True value ({estacion})')
        #plt.legend()

        #plt.show()

    return estacionScore

# COMMAND ----------

estacionScoreXGBoost = predict_estacionesXGBoost(X)

# COMMAND ----------

dfXGBoost = pd.DataFrame.from_dict(estacionScoreXGBoost, columns=['Modelo','MAE','Diferencia del Primer Dia','Error Absoluto del Primer Dia'],orient='index')
dfXGBoost['Error Absoluto del Primer Dia'] = dfXGBoost['Error Absoluto del Primer Dia'].apply(lambda x: x*100)
dfXGBoost['Error Absoluto del Primer Dia'] = dfXGBoost['Error Absoluto del Primer Dia'].apply(lambda x: float("{:.2f}".format(x)))
dfXGBoost['MAE'] = dfXGBoost['MAE'].apply(lambda x: float("{:.2f}".format(x)))
dfXGBoost['Diferencia del Primer Dia'] = dfXGBoost['Diferencia del Primer Dia'].apply(lambda x: float("{:.2f}".format(x)))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ### **Neural Networks**

# COMMAND ----------

seq_len = 30

def split_into_sequences(data, seq_len):
    n_seq = len(data) - seq_len + 1
    return np.array([data[i:(i+seq_len)] for i in range(n_seq)])

def get_train_test_sets(data, seq_len, train_frac):
    sequences = split_into_sequences(data, seq_len)
    n_train = int(sequences.shape[0] * train_frac)
    X_train = sequences[:n_train, :-1, :]
    y_train = sequences[:n_train, -1, :]
    X_test = sequences[n_train:, :-1, :]
    y_test = sequences[n_train:, -1, :]
    return X_train, y_train, X_test, y_test

# COMMAND ----------

from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Sequential

# fraction of the input to drop; helps prevent overfitting
dropout = 0.2
window_size = seq_len - 1

# build a 3-layer LSTM RNN
model = keras.Sequential()

model.add(
    LSTM(window_size, return_sequences=True, 
         input_shape=(window_size, x_train.shape[-1]))
)

model.add(Dropout(rate=dropout))
# Bidirectional allows for training of sequence data forwards and backwards
model.add(
    Bidirectional(LSTM((window_size * 2), return_sequences=True)
)) 

model.add(Dropout(rate=dropout))
model.add(
    Bidirectional(LSTM(window_size, return_sequences=False))
) 

model.add(Dense(units=1))
# linear activation function: activation is proportional to the input
model.add(Activation('linear'))

# COMMAND ----------

def MAE(trueArray,predArray):
    error = []
    for i,val in enumerate(predArray):
        try:
            error.append(abs(predArray[0][i] - trueArray[0][i]))
        except:
            error.append(-1)

    
    mae = float(sum(error))/len(predArray)

    return mae

# COMMAND ----------

MAE(y_test_orig,y_pred_orig)

# COMMAND ----------

def predict_estacionesNN(df,seq_len=30):

    estacionScore = {}
    for estacion in df['Nombre'].unique():
        print(f'PREDICCION PARA {estacion}')
        prophetdf = df[df['Nombre'] == estacion].reset_index(drop=True).sort_values(by='Fecha')[['VolumenM3']].dropna()

        scaler = MinMaxScaler()

        
    
        try:
            prophetdf = prophetdf['VolumenM3'].values.reshape(-1,1)
            scalerdf = scaler.fit_transform(prophetdf)


            x_train, y_train, x_test, y_test = get_train_test_sets(scalerdf, seq_len, train_frac=0.9)
            # fraction of the input to drop; helps prevent overfitting
            dropout = 0.2
            window_size = seq_len - 1

            # build a 3-layer LSTM RNN
            model = keras.Sequential()

            model.add(
                LSTM(window_size, return_sequences=True, 
                    input_shape=(window_size, x_train.shape[-1]))
            )

            model.add(Dropout(rate=dropout))
            # Bidirectional allows for training of sequence data forwards and backwards
            model.add(
                Bidirectional(LSTM((window_size * 2), return_sequences=True)
            )) 

            model.add(Dropout(rate=dropout))
            model.add(
                Bidirectional(LSTM(window_size, return_sequences=False))
            ) 

            model.add(Dense(units=1))
            # linear activation function: activation is proportional to the input
            model.add(Activation('linear'))

            batch_size = 16

            model.compile(
                loss='mean_squared_error',
                optimizer='adam'
            )

            history = model.fit(
                x_train,
                y_train,
                epochs=120,
                batch_size=batch_size,
                shuffle=False,
                validation_split=0.2
            )

            y_pred = model.predict(x_test)

            # invert the scaler to get the absolute price data
            y_test_orig = scaler.inverse_transform(y_test)
            y_pred_orig = scaler.inverse_transform(y_pred)

            
            try:
                #error = MAE(y_test_orig,y_pred_orig)
                errorDelDiaSiguiente = abs(float(y_test_orig[0][0] - y_pred_orig[0][0]))
                errorAbsolutoAlDiaSiguiente = abs((y_test_orig[0][0] - y_pred_orig[0][0])/y_pred_orig[0][0])

                #print(f'El MAE: {error} m3')
                print(f'El diferencia del dia siguiente es: {errorDelDiaSiguiente} m3')
                print(f'El error absoluto al dia siguiente es: {errorAbsolutoAlDiaSiguiente:.2f}%')

                estacionScore[estacion] = ['Bidirectional RNN',-1,errorDelDiaSiguiente,errorAbsolutoAlDiaSiguiente]
            except Exception as e:
                print('f#### ERROR EN ESTACION {estacion} ###########')
                print(e)


            # plots of prediction against actual data
            plt.plot(y_test_orig, label='Actual Consumption', color='orange')
            plt.plot(y_pred_orig, label='Predicted Consumption', color='green')

            plt.title('Gas consumption prediction')
            plt.xlabel('Days')
            plt.ylabel('Volume (m3)')
            plt.legend(loc='best')

            plt.show()

        except Exception as e:
            print(estacion)
            print(e)
            

        
        
    return estacionScore

# COMMAND ----------

NNdict = predict_estacionesNN(X)

# COMMAND ----------

prophetdf = X[X['Nombre'] == 'CARACOLI'].reset_index(drop=True).sort_values(by='Fecha')[['VolumenM3']]
scaler = MinMaxScaler()

prophetdf = prophetdf['VolumenM3'].values.reshape(-1,1)
scalerdf = scaler.fit_transform(prophetdf)

# COMMAND ----------

seq_len = 30

def split_into_sequences(data, seq_len):
    n_seq = len(data) - seq_len + 1
    return np.array([data[i:(i+seq_len)] for i in range(n_seq)])

def get_train_test_sets(data, seq_len, train_frac):
    sequences = split_into_sequences(data, seq_len)
    n_train = int(sequences.shape[0] * train_frac)
    X_train = sequences[:n_train, :-1, :]
    y_train = sequences[:n_train, -1, :]
    X_test = sequences[n_train:, :-1, :]
    y_test = sequences[n_train:, -1, :]
    return X_train, y_train, X_test, y_test

x_train, y_train, x_test, y_test = get_train_test_sets(scalerdf, seq_len, train_frac=0.9)

# COMMAND ----------

from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Sequential

# COMMAND ----------

from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Sequential

# fraction of the input to drop; helps prevent overfitting
dropout = 0.2
window_size = seq_len - 1

# build a 3-layer LSTM RNN
model = keras.Sequential()

model.add(
    LSTM(window_size, return_sequences=True, 
         input_shape=(window_size, x_train.shape[-1]))
)

model.add(Dropout(rate=dropout))
# Bidirectional allows for training of sequence data forwards and backwards
model.add(
    Bidirectional(LSTM((window_size * 2), return_sequences=True)
)) 

model.add(Dropout(rate=dropout))
model.add(
    Bidirectional(LSTM(window_size, return_sequences=False))
) 

model.add(Dense(units=1))
# linear activation function: activation is proportional to the input
model.add(Activation('linear'))

# COMMAND ----------

batch_size = 16

model.compile(
    loss='mean_squared_error',
    optimizer='adam'
)

history = model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=batch_size,
    shuffle=False,
    validation_split=0.2
)

# COMMAND ----------

y_pred = model.predict(x_test)

# invert the scaler to get the absolute price data
y_test_orig = scaler.inverse_transform(y_test)
y_pred_orig = scaler.inverse_transform(y_pred)

# plots of prediction against actual data
plt.plot(, label='Actual Consumption', color='orange')
plt.plot(y_pred_orig, label='Predicted Consumption', color='green')
 
plt.title('Gas consumption prediction')
plt.xlabel('Days')
plt.ylabel('Volume (m3)')
plt.legend(loc='best')

plt.show();


# COMMAND ----------

dfNN = pd.DataFrame.from_dict(NNdict, columns=['Modelo','MAE','Diferencia del Primer Dia','Error Absoluto del Primer Dia'],orient='index')
dfNN['Error Absoluto del Primer Dia'] = dfNN['Error Absoluto del Primer Dia'].apply(lambda x: x*100)
dfNN['Error Absoluto del Primer Dia'] = dfNN['Error Absoluto del Primer Dia'].apply(lambda x: float("{:.2f}".format(x)))
dfNN['MAE'] = dfNN['MAE'].apply(lambda x: float("{:.2f}".format(x)))
dfNN['Diferencia del Primer Dia'] = dfNN['Diferencia del Primer Dia'].apply(lambda x: float("{:.2f}".format(x)))

# COMMAND ----------

dfNN

# COMMAND ----------

y_pred_orig[0][0]

# COMMAND ----------

y_test_orig[0][0] - y_pred_orig[0][0]

# COMMAND ----------

scalerdf

# COMMAND ----------

results = pd.concat([df,dfARIMA,dfXGBoost,dfNN],axis=0).reset_index()
results = results.rename(columns={'index':'Nombre',
                                'Diferencia del Primer Dia':'DiferenciaDelPrimerDia',
                                'Error Absoluto del Primer Dia':'ErrorAbsolutoDelPrimerDia'})

# COMMAND ----------

results

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

schema = StructType([
    StructField("Nombre", StringType(), True),
    StructField("Modelo", StringType(), True),
    StructField("MAE", StringType(), True),
    StructField("DiferenciaDelPrimerDia", StringType(), True),
    StructField("ErrorAbsolutoDelPrimerDia", StringType(), True)
    ])
df = spark.createDataFrame(results, schema = schema)
df.write \
.format("com.databricks.spark.sqldw") \
.option("url", sqlDwUrl) \
.option("forwardSparkAzureStorageCredentials", "true") \
.option("dbTable", "ComercializacionML.ResultadosModelos") \
.option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
.mode("overwrite") \
.save()

# COMMAND ----------



# COMMAND ----------

import time
time.sleep(3600)
time.sleep(3600)
time.sleep(3600)

# COMMAND ----------

time.sleep(3600)

# COMMAND ----------

time.sleep(4000)

# COMMAND ----------

len(NNdict.values())

# COMMAND ----------

results[(results['ErrorAbsolutoDelPrimerDia'] <= 10) & (results['Modelo'] == 'ARIMA')]

# COMMAND ----------


