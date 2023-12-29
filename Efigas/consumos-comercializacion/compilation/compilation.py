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
tomorrow =  (datetime.now(timezone(timedelta(hours=-5), 'EST')) + timedelta(1))
tomorrow_str = (datetime.now(timezone(timedelta(hours=-5), 'EST')) + timedelta(1)).strftime("%Y-%m-%d")
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

mm = DeltaTable.forName(spark, f"analiticaefg.comercializacion.predicciones_mediamovil").toDF()
prophet = DeltaTable.forName(spark, f"analiticaefg.comercializacion.predicciones_prophet").toDF()
rnn = DeltaTable.forName(spark, f"analiticaefg.comercializacion.predicciones_rnn").toDF()
#estado = DeltaTable.forName(spark, "analiticaefg.comercializacion.estado").toDF()

#t1 = estado.groupBy("estacion").agg(pySparkMax("fecharegistro").alias("latest_fecharegistro"))
#estado_join = t1.alias('t1').join(estado.alias('e'), (t1.estacion == estado.estacion) & (t1.latest_fecharegistro == estado.fecharegistro),'inner').select('t1.estacion','e.estado','t1.latest_fecharegistro')
#results = insumo.alias('i').join(estado_join.alias('e'), (insumo.estacion == estado_join.estacion),'inner').select("i.estacion", "i.fecha", "i.volumenm3", "i.festivos", "i.diadesemana", "i.volumen_corregido", "residuals", "e.estado","e.latest_fecharegistro").where(f'e.estado = "Activa"').orderBy('i.estacion','i.fecha')

mm_df = mm.toPandas()
prophet_df = prophet.toPandas()
rnn_df = rnn.toPandas()

# COMMAND ----------

rnn_df[rnn_df['fecha'] == tomorrow.date()]

# COMMAND ----------

def choose_best_model(mm=mm_df,
                      prophet=prophet_df,
                      rnn=rnn_df,
                      date=tomorrow):
    mm = mm[mm['estado'].isin(['Nueva','Inactiva'])]
    prophet = prophet.copy()
    rnn = rnn.copy()

    df = pd.concat([prophet,rnn,mm],axis=0)
    df = df[df['fecha'] == date.date()]
    X = df.groupby('estacion').agg({'error_absoluto':'min'}).reset_index()

    best_score = X.merge(df,on=['estacion','error_absoluto'],how='inner')
    result = pd.concat([best_score,mm],axis=0)

    # Seleccionar solo la primera estacion si hay duplicados
    results = pd.DataFrame(columns=result.columns)

    for estacion in result['estacion'].unique():
        result_unique = result[result['estacion'] == estacion].head(1)

        results = pd.concat([results, result_unique],axis=0)
        

    return results


# COMMAND ----------

result = choose_best_model(mm=mm_df,
                      prophet=prophet_df,
                      rnn=rnn_df)

# COMMAND ----------

result['estacion'].value_counts()

# COMMAND ----------

result

# COMMAND ----------

result = result[['estacion','fecha','modelo','predicciones']]
result['fecha'] = pd.to_datetime(result['fecha'])

# COMMAND ----------

schema = StructType([
    StructField("estacion", StringType(), True),
    StructField("fecha", DateType(), True),
    StructField("modelo", StringType(), True),
    StructField("prediccion", FloatType(), True)
    ])
df = spark.createDataFrame(result, schema = schema)

deltaTable = DeltaTable.forName(spark, 'analiticaefg.comercializacion.predicciones')

deltaTable.alias("t").merge(
    df.alias("s"),
    "t.estacion = s.estacion AND t.fecha = s.fecha"
).whenNotMatchedInsertAll().execute()
