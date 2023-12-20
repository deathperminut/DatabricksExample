# Databricks notebook source
import os
import pandas as pd
import numpy as np
import pickle as pkl
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, DateType
from pyspark.sql import SparkSession
from pyspark.sql.functions import max as pySparkMax
from pyspark.sql.functions import col
from prophet import Prophet
from delta.tables import DeltaTable
import random
import time
from datetime import date,datetime,timedelta,timezone
today = datetime.now(timezone(timedelta(hours=-5), 'EST'))
yesterday = (datetime.now(timezone(timedelta(hours=-5), 'EST')) - timedelta(1)).strftime("%Y-%m-%d")
tomorrow = (datetime.now(timezone(timedelta(hours=-5), 'EST')) + timedelta(1)).strftime("%Y-%m-%d")
two_days_prior = (datetime.now(timezone(timedelta(hours=-5), 'EST')) - timedelta(2)).strftime("%Y-%m-%d")
today_dt = today.strftime("%Y-%m-%d")
one_year_ago = (datetime.now(timezone(timedelta(hours=-5), 'EST')) - timedelta(days=365)).strftime("%Y-%m-%d")
two_years_ago = (datetime.now(timezone(timedelta(hours=-5), 'EST')) - timedelta(days=730)).strftime("%Y-%m-%d")


import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

storage_account_name = dbutils.secrets.get(scope='efigas', key='bs-name')
blob_container = dbutils.secrets.get(scope='efigas', key='bs-container')
blob_storage = storage_account_name + ".blob.core.windows.net"
config_key = "fs.azure.account.key."+storage_account_name+".blob.core.windows.net"
blob_access_key = dbutils.secrets.get(scope='efigas', key='bs-access-key')
spark.conf.set(config_key, blob_access_key)

# COMMAND ----------

insumo = DeltaTable.forName(spark, f"analiticaefg.comercializacion.insumo").toDF()
estado = DeltaTable.forName(spark, "analiticaefg.comercializacion.estado").toDF()

t1 = estado.groupBy("estacion").agg(pySparkMax("fecharegistro").alias("latest_fecharegistro"))

estado_join = t1.alias('t1').join(estado.alias('e'), (t1.estacion == estado.estacion) & (t1.latest_fecharegistro == estado.fecharegistro),'inner').select('t1.estacion','e.estado','t1.latest_fecharegistro')

results = insumo.alias('i').join(estado_join.alias('e'), (insumo.estacion == estado_join.estacion),'inner').select("i.estacion", "i.fecha", "i.volumenm3", "i.festivos", "i.diadesemana", "i.volumen_corregido", "residuals", "e.estado","e.latest_fecharegistro").where(f'e.estado = "Activa"').orderBy('i.estacion','i.fecha')

results = results.toPandas()

# COMMAND ----------

def predict_estaciones(df, two_years_ago, two_days_prior,today):
    # List to store station-wise scores
    estacion_scores = []

    # Filter data based on the given time frame
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df[df['fecha'] >= two_years_ago]
    predicciones = 0
    
    # Iterate through unique stations (top 10)
    for estacion in df['estacion'].unique():
        print(f'PREDICTION FOR {estacion}')
        
        # Prepare data for Prophet
        prophetdf = df[df['estacion'] == estacion].reset_index(drop=True).sort_values(by='fecha')[['fecha', 'volumen_corregido']]
        prophetdf.columns = ['ds', 'y']
        prophetdf['ds'] = pd.to_datetime(prophetdf['ds'])

        prophetTrain = prophetdf[prophetdf['ds'] < two_days_prior]
        prophetTest = prophetdf[prophetdf['ds'] >= two_days_prior]

        try:
            # Modeling and forecasting
            model = Prophet()
            model.fit(prophetTrain)

            future_dates = model.make_future_dataframe(periods=prophetTest.shape[0], freq='d')
            forecast = model.predict(future_dates)
            
            # Post-processing forecast results
            forecast['ds'] = pd.to_datetime(forecast['ds'])
            newForecast = forecast[forecast['ds'] >= two_days_prior][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            newForecast['yreal'] = prophetTest['y'].tolist()
            newForecast['error'] = abs(newForecast['yreal'] - newForecast['yhat'])
            newForecast['yhat'] = newForecast['yhat'].apply(lambda x: 0 if x < 0 else x)
            predicciones = newForecast.tail(1)['yhat'].values[0]

            # Calculating errors
            error = newForecast['error'].mean()
            errorDelDiaSiguiente = newForecast.iloc[0]['error']
            errorAbsolutoAlDiaSiguiente = abs((newForecast.iloc[0]['yreal'] - newForecast.iloc[0]['yhat']) / newForecast.iloc[0]['yhat']) * 100

            # Printing and storing scores
            print(f'MAE: {error:.2f} m3')
            print(f'Error for the next day: {errorDelDiaSiguiente:.2f} m3')
            print(f'Absolute error for the next day: {errorAbsolutoAlDiaSiguiente:.2f}%')

            estacion_scores.append({
                'estacion': estacion,
                'modelo': 'Prophet Forecasting',
                'mae': error,
                'fecha': today,
                'predicciones': predicciones,
                'error': errorDelDiaSiguiente,
                'error_absoluto': errorAbsolutoAlDiaSiguiente,
                'estado': 'Activa'
            })
        except Exception as e:
            estacion_scores.append({
                'estacion': estacion,
                'modelo': 'Prophet Forecasting',
                'mae': -1,
                'fecha': today,
                'predicciones': predicciones,
                'error': -1,
                'error_absoluto': -1,
                'estado': 'Activa'
            })

    return pd.DataFrame(estacion_scores)

# COMMAND ----------

prophet_results = predict_estaciones(results,two_years_ago, two_days_prior,today)

# COMMAND ----------

predicciones = prophet_results[['estacion','fecha','predicciones','error_absoluto','modelo','estado']]
predicciones

# COMMAND ----------

schema = StructType([
    StructField("estacion", StringType(), True),
    StructField("fecha", DateType(), True),
    StructField("predicciones", FloatType(), True),
    StructField("error_absoluto", FloatType(), True),
    StructField("modelo", StringType(), True),
    StructField("estado", StringType(), True)
    ])
df = spark.createDataFrame(predicciones, schema = schema)

deltaTable = DeltaTable.forName(spark, 'analiticaefg.comercializacion.predicciones_prophet')

deltaTable.alias("t").merge(
    df.alias("s"),
    "t.estacion = s.estacion AND t.fecha = s.fecha"
).whenNotMatchedInsertAll().execute()

# COMMAND ----------


