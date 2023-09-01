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
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM, BatchNormalization
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from datetime import date,datetime,timedelta
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

ordinalNombre = {}
ordinalTipoUsuario = {}
for i,est in enumerate(rawData['Nombre'].unique()):
    ordinalNombre[est] = i

for i,tipo in enumerate(rawData['TipoUsuario'].unique()):
    ordinalTipoUsuario[tipo] = i


ordinalTipoUsuario

# COMMAND ----------

rawData[rawData['Fecha'] == rawData['Fecha'].max()]

# COMMAND ----------

def onehot_encode(df, column):
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df, dummies], axis=1)
    return df

# COMMAND ----------

def process_inputs(df,ordinalNombre=ordinalNombre,ordinalTipoUsuario=ordinalTipoUsuario):
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
    df['DiaDeSemana'] = df['Fecha'].apply(lambda x: x.dayofweek)

    for i,festivo in enumerate(df['Festivos']):
        if festivo == 1:
            df['DiaDeSemana'][i] = 8
        else:
            pass

    df['OrdinalNombre'] = df['Nombre'].replace(ordinalNombre).astype(int)
    df['OrdinalTipoUsuario'] = df['TipoUsuario'].replace(ordinalTipoUsuario).astype(int)
    
    # Reemplazar valores que salen como 0E-8 a 0.
    df['VolumenM3'] = df['VolumenM3'].replace({0E-8:0})
    
    
    uniqueDispositivos = df['IdDispositivo'].unique()
    
    newDisp = pd.DataFrame(columns=['IdDispositivo', 'Nombre', 'TipoUsuario', 'Fecha', 'VolumenM3','VolumenCumSum','DiaDeSemana'])
    
    for disp in uniqueDispositivos:
        dfDispSum = df[df['IdDispositivo'] == disp].sort_values(by='Fecha')
        try:
            dfDispSum['VolumenCumSum'] = dfDispSum['VolumenM3'].cumsum().astype('float')
        except Exception as e:
            print(f'{disp}: {e}')
        
        newDisp = pd.concat([dfDispSum,newDisp],axis=0)
    
    newDisp['VolumenM3'] = newDisp['VolumenM3'].astype('float')
    newDisp['Festivos'] = newDisp['Festivos'].astype('int')
    

    return newDisp
    

# COMMAND ----------

X = process_inputs(rawData)

# COMMAND ----------

X['OrdinalNombre'].value_counts()

# COMMAND ----------

sampleEstaciones = list(rawData.dropna()['Nombre'].value_counts().sort_values(ascending=False)[:-1].index)

newSampleEstaciones = []

for est in sampleEstaciones:
    newSampleEstaciones.append(f'Nombre_{est}')

#colsNombre = [x for x in X.columns if x.startswith('Nombre_')]
colsOG = ['VolumenM3','DiaDeSemana','TipoUsuario']
colsTipoUsuario = [x for x in X.columns if x.startswith('TipoUsuario_')]

cols = newSampleEstaciones + colsOG + colsTipoUsuario

# COMMAND ----------

 yesterday = datetime.now() - timedelta(1)
 yesterdayStr = datetime.strftime(yesterday, '%Y-%m-%d')
 yesterdayStr

# COMMAND ----------



# COMMAND ----------

(today - yesterday).days

# COMMAND ----------

estaciones = X['Nombre'].unique()

yesterday = datetime.now() - timedelta(1)
yesterdayStr = datetime.strftime(yesterday, '%Y-%m-%d')
empDF = pd.DataFrame(columns=X.columns)
estacionesABorrar = []

for estacion in estaciones:
    
    dfEstacion = X[X['Nombre'] == estacion]
    dfEstacion = dfEstacion[dfEstacion['Fecha'] <= yesterdayStr]

    fechaMax = dfEstacion['Fecha'].max()
    #fechaMaxDT = datetime.strptime(fechaMax, '%Y-%m-%d')
    fechaDif = (yesterday - fechaMax).days
    if fechaDif > 0:
        print(f'Estacion: {estacion} \n Dias desde el ultimo reporte: {fechaDif}')
        if fechaDif >= 60:
            estacionesABorrar.append(estacion)
        else:
            for date in range(fechaDif):
                

    
    empDF = pd.concat([dfEstacion,empDF],axis=0)



# COMMAND ----------

for i in range(50):
    print(datetime.strftime(today - timedelta(i), '%Y-%m-%d'))
    

# COMMAND ----------

sampleEstaciones = list(rawData.dropna()['Nombre'].value_counts().sort_values(ascending=False)[:-1].index)
newX = X[X['Nombre'].isin(sampleEstaciones)]

# COMMAND ----------

cols = ['OrdinalNombre','VolumenM3','DiaDeSemana','OrdinalTipoUsuario']

# COMMAND ----------

prophetTrain = newX[newX['Fecha'] < '2023-04-01']

prophetTest = newX[newX['Fecha'] >= '2023-04-01']

# COMMAND ----------

prophetTrain

# COMMAND ----------

scaler = MinMaxScaler()

prophetdf = prophetTrain[cols]
scalerdf = scaler.fit_transform(prophetdf)

# COMMAND ----------

prophetdfTest = prophetTest[cols]
scalerdfTest = scaler.transform(prophetdfTest)

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
    return X_train, y_train
    
x_train, y_train = get_train_test_sets(scalerdf, seq_len, train_frac=1)

# COMMAND ----------

x_test, y_test = get_train_test_sets(scalerdfTest, seq_len, train_frac=1)

# COMMAND ----------

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
model.add(BatchNormalization())
model.add(Dropout(rate=dropout))
model.add(
    Bidirectional(LSTM(window_size, return_sequences=True))
) 
model.add(Dropout(rate=0.5))
model.add(
    Bidirectional(LSTM(window_size, return_sequences=False))
) 

model.add(Dense(units=1))
# linear activation function: activation is proportional to the input
model.add(Activation('linear'))

# COMMAND ----------

x_train

# COMMAND ----------

batch_size = 16

model.compile(
    loss='mean_squared_error',
    optimizer='rmsprop'
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

print(history.history.keys())
#  "Accuracy"
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"

# COMMAND ----------

model.predict(x_test).shape

# COMMAND ----------

_1,_2,_3,_4 = y_train.T
_2

# COMMAND ----------

y_trainOrdinalNombre, y_trainVolumenM3, y_trainDiaDeSemana, y_trainOrdinalTipoUsuario = y_train.T

batch_size = 16

model.compile(
    loss='mean_squared_error',
    optimizer='adam'
)

history = model.fit(
    x_train,
    y_trainVolumenM3,
    epochs=30,
    batch_size=batch_size,
    shuffle=False,
    validation_split=0.2
)

# COMMAND ----------

print(history.history.keys())
#  "Accuracy"
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"

# COMMAND ----------


