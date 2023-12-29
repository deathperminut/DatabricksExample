# Databricks notebook source
import os
import pandas as pd
import numpy as np
import pickle as pkl
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, DateType
from pyspark.sql import SparkSession
from pyspark.sql.functions import max as pySparkMax
from pyspark.sql.functions import col
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

insumo = DeltaTable.forName(spark, "analiticaefg.comercializacion.insumo").toDF()
estado = DeltaTable.forName(spark, "analiticaefg.comercializacion.estado").toDF()

t1 = estado.groupBy("estacion").agg(pySparkMax("fecharegistro").alias("latest_fecharegistro"))

estado_join = t1.alias('t1').join(estado.alias('e'), (t1.estacion == estado.estacion) & (t1.latest_fecharegistro == estado.fecharegistro),'inner').select('t1.estacion','e.estado','t1.latest_fecharegistro')

results = insumo.alias('i').join(estado_join.alias('e'), (insumo.estacion == estado_join.estacion),'inner').select("i.estacion", "i.fecha", "i.volumenm3", "i.festivos", "i.diadesemana", "i.volumen_corregido", "residuals", "e.estado","e.latest_fecharegistro").where('e.estado in ("Nueva","Inactiva","Activa")').orderBy('i.estacion','i.fecha')

results = results.toPandas()

# COMMAND ----------

nueva = results[results['estado'] == 'Nueva']
inactiva = results[results['estado'] == 'Inactiva']
activa = results[results['estado'] == 'Activa']

print(f'Tamano de Nueva: {nueva.shape}.\nEstaciones en Nueva: {len(nueva["estacion"].unique())}')
print(f'Tamano de Inactiva: {inactiva.shape}.\nEstaciones en Inactiva: {len(inactiva["estacion"].unique())}')
print(f'Tamano de Nueva: {activa.shape}.\nEstaciones en Nueva: {len(activa["estacion"].unique())}')

# COMMAND ----------

def make_predictions_nueva(df,column,fecha):
    df = df.copy()

    pred_dict = {}
    estaciones = []
    predicciones = []
    fechas = []

    for estacion in df['estacion'].unique():
        _ = df[df['estacion'] == estacion]

        latest_mean = _[column].mean()

        estaciones.append(estacion)
        predicciones.append(latest_mean)
        fechas.append(fecha)
    
    pred_dict['estacion'] = estaciones
    pred_dict['fecha'] = fechas
    pred_dict['predicciones'] = predicciones
    pred_dict['modelo'] = ['Media Movil']*len(df['estacion'].unique())
    pred_dict['estado'] = ['Nueva']*len(df['estacion'].unique())

    predicciones_df = pd.DataFrame(pred_dict)
    predicciones_df['fecha'] = pd.to_datetime(predicciones_df['fecha'])

    return predicciones_df

# COMMAND ----------

pred_nueva = make_predictions_nueva(df=nueva,
                                    column='volumen_corregido',
                                    fecha=tomorrow)

pred_nueva.head(10)

# COMMAND ----------

def make_predictions_inactiva(df,fecha):
    df = df.copy()

    pred_dict = {}
    estaciones = []
    predicciones = []
    fechas = []

    for estacion in df['estacion'].unique():
        estaciones.append(estacion)
        predicciones.append(0)
        fechas.append(fecha)
    
    pred_dict['estacion'] = estaciones
    pred_dict['fecha'] = fechas
    pred_dict['predicciones'] = predicciones
    pred_dict['modelo'] = ['Media Movil']*len(df['estacion'].unique())
    pred_dict['estado'] = ['Inactiva']*len(df['estacion'].unique())

    predicciones_df = pd.DataFrame(pred_dict)
    predicciones_df['fecha'] = pd.to_datetime(predicciones_df['fecha'])

    

    return predicciones_df

# COMMAND ----------

pred_inactiva = make_predictions_inactiva(df=inactiva,
                                          fecha=tomorrow)

pred_inactiva.head(10)

# COMMAND ----------

def make_predictions_activa(df,column,fecha,dias_mean=10):
    df = df.copy()

    pred_dict = {}
    estaciones = []
    predicciones = []
    fechas = []

    for estacion in df['estacion'].unique():
        _ = df[df['estacion'] == estacion]

        latest_mean = _.tail(dias_mean)[column].mean()

        estaciones.append(estacion)
        predicciones.append(latest_mean)
        fechas.append(fecha)
    
    pred_dict['estacion'] = estaciones
    pred_dict['fecha'] = fechas
    pred_dict['predicciones'] = predicciones
    pred_dict['modelo'] = ['Media Movil']*len(df['estacion'].unique())
    pred_dict['estado'] = ['Activa']*len(df['estacion'].unique())

    predicciones_df = pd.DataFrame(pred_dict)
    predicciones_df['fecha'] = pd.to_datetime(predicciones_df['fecha'])

    return predicciones_df

# COMMAND ----------

pred_activa = make_predictions_activa(df=activa,
                                      column='volumen_corregido',
                                      fecha=tomorrow)

pred_activa.head(10)

# COMMAND ----------

predicciones = pd.concat([pred_nueva,pred_inactiva,pred_activa],axis=0)
predicciones

# COMMAND ----------

schema = StructType([
    StructField("estacion", StringType(), True),
    StructField("fecha", DateType(), True),
    StructField("predicciones", FloatType(), True),
    StructField("modelo", StringType(), True),
    StructField("estado", StringType(), True)
    ])
df = spark.createDataFrame(predicciones, schema = schema)

deltaTable = DeltaTable.forName(spark, 'analiticaefg.comercializacion.predicciones_mediamovil')

deltaTable.alias("t").merge(
    df.alias("s"),
    "t.estacion = s.estacion AND t.fecha = s.fecha"
).whenNotMatchedInsertAll().execute()

# COMMAND ----------


