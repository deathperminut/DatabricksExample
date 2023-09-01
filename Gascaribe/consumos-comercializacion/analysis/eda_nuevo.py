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
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet
from datetime import date,datetime
import holidays
today = datetime.now()
today_dt = today.strftime("%d-%m-%Y")

import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

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

def process_inputs(df,tipoUsuario):
    df = df.copy()
    
    # Filtrar por tipo de usuario
    df = df[df['TipoUsuario'] == tipoUsuario]
    
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

X = process_inputs(rawData,'ESTACION')
X.shape

# COMMAND ----------

def sample_tendency(df,normalize=False):


    uniqueEstaciones = df['Nombre'].unique()
    similarEstaciones = []
    # Creates the new DataFrame with two estaciones

    for i,est in enumerate(uniqueEstaciones):
        print(f"ANALISIS ENTRE: {uniqueEstaciones[i]} y {uniqueEstaciones[i+1]}")
        sampleX = pd.DataFrame(columns=['IdDispositivo', 'Nombre', 'TipoUsuario', 'Red', 'Fecha', 'VolumenM3','VolumenCumSum'])

        _1 = df[df['Nombre'] == uniqueEstaciones[i]]
        _2 = df[df['Nombre'] == uniqueEstaciones[i+1]]

        print(f"Iteraciones: [{uniqueEstaciones[i]} {(i)},{uniqueEstaciones[i+1]}{(i+1)}]")

        sampleX = pd.concat([_1,_2,sampleX],axis=0)

        dfValuesSampleX1 = pd.DataFrame(columns=['1','2'])
        dfValuesSampleX2 = pd.DataFrame(columns=['2','1'])

        slicedSampleX = sampleX.groupby('Nombre').agg({'VolumenM3':['count']})
        slicedSampleX.columns = ['_'.join(col) for col in slicedSampleX.columns.values]
        slicedSampleX = slicedSampleX.sort_values(by='VolumenM3_count',ascending=True).reset_index()
        minRows = slicedSampleX.iloc[0]['VolumenM3_count']

        if normalize:
            list1 = list(_1['VolumenCumSum'])
            list2 = list(_2['VolumenCumSum'])

            minList1 = min(list1)
            minList2 = min(list2)
            maxList1 = max(list1)
            maxList2 = max(list2)

            normList1 = [(x - minList1)/(maxList1 - minList1) for x in list1]
            normList2 = [(x - minList2)/(maxList2 - minList2) for x in list2]

            dfValuesSampleX1['1'] = normList1[:minRows]
            dfValuesSampleX1['2'] = normList2[:minRows]
            dfValuesSampleX2['1'] = normList1[:minRows]
            dfValuesSampleX2['2'] = normList2[:minRows]
        
        else:
            dfValuesSampleX1['1'] = list(_1['VolumenCumSum'])[:minRows]
            dfValuesSampleX1['2'] = list(_2['VolumenCumSum'])[:minRows]
            dfValuesSampleX2['1'] = list(_1['VolumenCumSum'])[:minRows]
            dfValuesSampleX2['2'] = list(_2['VolumenCumSum'])[:minRows]


        #print(sampleX.head())
        print(f"Shape of the sampled dataframe:                                                 {dfValuesSampleX1.shape}")
        print(f"List of unique estaciones in the sampled dataframe:                             {sampleX['Nombre'].unique()}")
        print(f"List of unique Dispositivos in the sampled dataframe:                           {sampleX['IdDispositivo'].unique()}")
        
        
        

        try:
            grang = grangercausalitytests(
                dfValuesSampleX1.head(minRows)
                ,maxlag=[3])

            grang = grangercausalitytests(
                dfValuesSampleX2.head(minRows)
                ,maxlag=[3])
            
            plt.plot(dfValuesSampleX1['1'])
            plt.plot(dfValuesSampleX1['2'])
            plt.show()
        except:
            print(sampleX['IdDispositivo'].unique())
    
    return grang

# COMMAND ----------

g = sample_tendency(X)

# COMMAND ----------

dict1 = {'1':X[X['Nombre'] == 'CHORRERA'].reset_index(drop=True).sort_values(by='VolumenCumSum',ascending=True)['VolumenCumSum'],
        '2':X[X['Nombre'] == 'CHORRERA'].reset_index(drop=True).sort_values(by='VolumenCumSum',ascending=True)['VolumenCumSum']}

_ = pd.DataFrame(dict1,columns=['1','2'])

_.tail()

# COMMAND ----------

grang = grangercausalitytests(_,maxlag=3)

# COMMAND ----------

def DTWDistance(s1, s2):
    DTW={}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return (DTW[len(s1)-1, len(s2)-1])**(1/2)

# COMMAND ----------

DTWDistance(X[X['Nombre'] == 'PIMSA63'].reset_index(drop=True).sort_values(by='VolumenCumSum',ascending=True)['VolumenM3'],X[X['Nombre'] == 'LAS FLORES 62'].reset_index(drop=True).sort_values(by='VolumenCumSum',ascending=True)['VolumenM3'])

# COMMAND ----------

import time

def DTWD(df):

    uniqueEstaciones = df['Nombre'].unique()
    similarEstaciones = {}
    # Creates the new DataFrame with two estaciones

    for i,est in enumerate(uniqueEstaciones[:-2]):
        for j,estacion in enumerate(uniqueEstaciones[i+1:]):

            start = time.time()

            estacionA = uniqueEstaciones[i]
            estacionB = uniqueEstaciones[j]

            print(f"ANALISIS ENTRE: {estacionA} y {estacionB}")

            listA = list(X[X['Nombre'] == estacionA].reset_index(drop=True).sort_values(by='VolumenCumSum',ascending=True)['VolumenM3'])
            listB = list(X[X['Nombre'] == estacionB].reset_index(drop=True).sort_values(by='VolumenCumSum',ascending=True)['VolumenM3'])

            dist = DTWDistance(listA,listB)

            similarEstaciones[f'{estacionA} - {estacionB}'] = dist

            end = time.time()
            a = end - start
            print(f'Elapsed time: {a:.2f} s')

    return similarEstaciones

# COMMAND ----------

s = DTWD(X)

# COMMAND ----------

s

# COMMAND ----------

sSorted = {k: v for k, v in sorted(s.items(), key=lambda item: item[1])}

# COMMAND ----------

sSorted

# COMMAND ----------

for i in X['Nombre'].unique():
    print(sSorted[f'{i} - {X["Nombre"].unique()[1]}'])

# COMMAND ----------

del sSorted['SANTA LUCIA - PENDALES']
del sSorted['PENDALES - PUERTO GIRALDO']

# COMMAND ----------

np.quantile(list(sSorted.values()),0.2)

# COMMAND ----------

def plot_estaciones(df,estacionA,estacionB):

    listA = list(X[X['Nombre'] == estacionA].reset_index(drop=True).sort_values(by='VolumenCumSum',ascending=True)['VolumenCumSum'])
    listB = list(X[X['Nombre'] == estacionB].reset_index(drop=True).sort_values(by='VolumenCumSum',ascending=True)['VolumenCumSum'])

    plt.figure(figsize=(15,8))

    plt.plot(listA)
    plt.plot(listB)

    plt.show()

# COMMAND ----------

plot_estaciones(X,'MOLINEROS','LAS FLORES')

# COMMAND ----------

estacionA = 'CANNON ESTACION'
estacionB = 'TAREL'

prophetdf = X[X['Nombre'] == estacionA].reset_index(drop=True).sort_values(by='Fecha')[['Fecha','VolumenM3']]
prophetdfB = X[X['Nombre'] == estacionB].reset_index(drop=True).sort_values(by='Fecha')[['Fecha','VolumenM3']]

prophetdf.columns = ['ds','y']
prophetdfB.columns = ['ds','y']

# COMMAND ----------

prophetTrain = prophetdf[prophetdf['ds'] < '2022-08-01']
prophetTest = prophetdf[prophetdf['ds'] >= '2022-08-01']

prophetTrainB = prophetdfB[prophetdfB['ds'] < '2022-08-01']
prophetTestB = prophetdfB[prophetdfB['ds'] >= '2022-08-01']

# COMMAND ----------



# COMMAND ----------

model = Prophet()
model.fit(prophetTrain)

future_dates = model.make_future_dataframe(periods=prophetTest.shape[0], freq='d')

forecast = model.predict(future_dates)

# COMMAND ----------

newForecast = forecast[forecast['ds'] >= '2022-08-01'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

newForecast['yreal'] = prophetTest['y']
newForecast['yreal_different'] = prophetTestB['y']

# COMMAND ----------

newForecast

# COMMAND ----------

plt.figure(figsize=(15,8))

plt.plot(newForecast['yhat'],label='Predicted')
plt.plot(newForecast['yreal'],label='True value (CANNON ESTACION)')
plt.plot(newForecast['yreal_different'],label='True value (TAREL)')
plt.legend()

plt.show()

# COMMAND ----------

model.plot(forecast, uncertainty=True)

# COMMAND ----------

def predict_estaciones(df,estacionA,estacionB):

    prophetdf = X[X['Nombre'] == estacionA].reset_index(drop=True).sort_values(by='Fecha')[['Fecha','VolumenM3']]
    prophetdfB = X[X['Nombre'] == estacionB].reset_index(drop=True).sort_values(by='Fecha')[['Fecha','VolumenM3']]

    prophetdf.columns = ['ds','y']
    prophetdfB.columns = ['ds','y']

    prophetTrain = prophetdf[prophetdf['ds'] < '2022-08-01']
    prophetTest = prophetdf[prophetdf['ds'] >= '2022-08-01']

    prophetTrainB = prophetdfB[prophetdfB['ds'] < '2022-08-01']
    prophetTestB = prophetdfB[prophetdfB['ds'] >= '2022-08-01']

    model = Prophet()
    model.fit(prophetTrain)

    future_dates = model.make_future_dataframe(periods=prophetTest.shape[0], freq='d')

    forecast = model.predict(future_dates)

    newForecast = forecast[forecast['ds'] >= '2022-08-01'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    newForecast['yreal'] = prophetTest['y']
    newForecast['yreal_different'] = prophetTestB['y']

    # Plot forecasts
    plt.figure(figsize=(15,8))

    plt.plot(newForecast['yhat_upper'],label='Predicted')
    plt.plot(newForecast['yreal'],label=f'True value ({estacionA})')
    plt.plot(newForecast['yreal_different'],label=f'True value ({estacionB})')
    plt.legend()

    plt.show()

# COMMAND ----------

sSorted

# COMMAND ----------

predict_estaciones(X,'BUENOS AIRES','USIACURI')

# COMMAND ----------

list(sSorted.keys())[9]

# COMMAND ----------

model

# COMMAND ----------

model1 = Prophet()

# COMMAND ----------

model1

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### **Extracting Seasonalities**

# COMMAND ----------

estaciones = []

# COMMAND ----------

estaciones = X['Nombre'].unique()

for estacion in estaciones[2:3]:
    

    fig, axs = plt.subplots(2, figsize=(12,8))
    plot_acf(X[X['Nombre'] == estacion]['VolumenCumSum'], ax=axs[0])
    plot_pacf(X[X['Nombre'] == estacion]['VolumenCumSum'], ax=axs[1])
    plt.show()

# COMMAND ----------

from statsmodels.tsa.statespace.arima import ARIMA

# Define the seasonal component of the time series
seasonal_period = 1

# Calculate the seasonal component of each time series
seasonal_components = df.groupby(X['Fecha'])['VolumenCumSum'].mean()

# Create a matrix with the seasonal components of each time series
X = np.array([seasonal_components.values for i in range(len(X))])

# Use K-means clustering to group the time series based on their seasonal patterns
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
labels = kmeans.labels_

# Visualize the time series grouped by cluster
fig, axs = plt.subplots(n_clusters, figsize=(12,8), sharex=True)
for i in range(n_clusters):
    axs[i].plot(df[labels==i]['GasConsumption'], label=f'Cluster {i}')
    axs[i].set_title(f'Cluster {i}')
    axs[i].legend()

# Build a SARIMAX model for each cluster
models = []
for i in range(n_clusters):
    cluster_df = df[labels==i]
    model = SARIMAX(cluster_df['GasConsumption'], order=(1,1,1), seasonal_order=(1,1,1,seasonal_period))
    models.append(model.fit())

# Make predictions for each cluster
predictions = []
for i in range(n_clusters):
    model = models[i]
    cluster_df = df[labels==i]
    start_date = cluster_df.index[-1] + pd.Timedelta(days=1)
    end_date = start_date + pd.Timedelta(days=7)
    pred = model.predict(start=start_date, end=end_date)
    predictions.append(pred)

# Combine the predictions for each cluster into a single forecast
forecast = pd.concat(predictions)

# Visualize the forecast
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(df['GasConsumption'], label='Actual')
ax.plot(forecast, label='Forecast')
ax.legend()


# COMMAND ----------

# Calculate the seasonal component of each time series
seasonal_components = df.groupby(X['Fecha'])['VolumenCumSum'].mean()

# Create a matrix with the seasonal components of each time series
Y = np.array([seasonal_components.values for i in range(len(X))])

# COMMAND ----------

def predict_estaciones(df):

    estacionScore = {}
    for estacion in df['Nombre'].unique():
        prophetdf = X[X['Nombre'] == estacion].reset_index(drop=True).sort_values(by='Fecha')[['Fecha','VolumenM3']]


        prophetdf.columns = ['ds','y']

        prophetTrain = prophetdf[prophetdf['ds'] < '2022-12-01']
        prophetTest = prophetdf[prophetdf['ds'] >= '2022-12-01']


        model = Prophet()
        model.fit(prophetTrain)

        future_dates = model.make_future_dataframe(periods=prophetTest.shape[0], freq='d')

        forecast = model.predict(future_dates)

        newForecast = forecast[forecast['ds'] >= '2022-12-01'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        newForecast['yreal'] = prophetTest['y']
        newForecast['error'] = abs(newForecast['yreal'] - newForecast['yhat'])
        newForecast['yhat'] = newForecast['yhat'].apply(lambda x: 0 if x < 0 else x)

        
        try:
            error = newForecast['error'].sum()/len(newForecast)
            errorDelDiaSiguiente = float(newForecast.head(1)['error'].values[0])/len(newForecast)
            errorAbsolutoAlDiaSiguiente = abs((newForecast.head(1)['yreal'].values[0] - newForecast.head(1)['yhat'].values[0])/newForecast.head(1)['yhat'].values[0])

            print(f'El MAE: {error} m3')
            print(f'El error del dia siguiente es: {errorDelDiaSiguiente} m3')
            print(f'El error absoluto al dia siguiente es: {errorAbsolutoAlDiaSiguiente:.2f}%')
        except:
            pass



        # Plot forecasts
        plt.figure(figsize=(15,8))

        plt.plot(newForecast['yhat'],label='Predicted')
        plt.plot(newForecast['yreal'],label=f'True value ({estacion})')
        plt.legend()

        plt.show()

# COMMAND ----------

predict_estaciones(X)

# COMMAND ----------

predict_estaciones(X)

# COMMAND ----------

X.head()

# COMMAND ----------

time.sleep(3600)

# COMMAND ----------


