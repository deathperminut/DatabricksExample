# Databricks notebook source
import os
import pandas as pd
import numpy as np
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, DateType
from pyspark.sql import SparkSession
from delta.tables import DeltaTable

import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

estacionesNoUtilizables = [45, 107]

# COMMAND ----------

dwDatabase = dbutils.secrets.get(scope='efigas', key='dwh-name')
dwServer = dbutils.secrets.get(scope='efigas', key='dwh-host')
dwUser = dbutils.secrets.get(scope='efigas', key='dwh-user')
dwPass = dbutils.secrets.get(scope='efigas', key='dwh-pass')
dwJdbcPort = dbutils.secrets.get(scope='efigas', key='dwh-port')
dwJdbcExtraOptions = ""
sqlDwUrl = "jdbc:sqlserver://" + dwServer + ".database.windows.net:" + dwJdbcPort + ";database=" + dwDatabase + ";user=" + dwUser + ";password=" + dwPass + ";" + dwJdbcExtraOptions
storage_account_name = dbutils.secrets.get(scope='efigas', key='bs-name')
blob_container = dbutils.secrets.get(scope='efigas', key='bs-container')
blob_storage = storage_account_name + ".blob.core.windows.net"
config_key = "fs.azure.account.key."+storage_account_name+".blob.core.windows.net"
blob_access_key = dbutils.secrets.get(scope='efigas', key='bs-access-key')
spark.conf.set(config_key, blob_access_key)

# COMMAND ----------

query = 'SELECT * FROM Scada.DimEstacionesComercializacion'
consumption_query = 'SELECT * FROM ComercializacionML.IngestaBricks'

df = spark.read \
    .format("com.databricks.spark.sqldw") \
    .option("url", sqlDwUrl) \
    .option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
    .option("forwardSparkAzureStorageCredentials", "true") \
    .option("maxStrLength", "1024" ) \
    .option("query", query) \
    .load()

consumption_df = spark.read \
    .format("com.databricks.spark.sqldw") \
    .option("url", sqlDwUrl) \
    .option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
    .option("forwardSparkAzureStorageCredentials", "true") \
    .option("maxStrLength", "1024" ) \
    .option("query", consumption_query) \
    .load()

rawData = df.toPandas()
rawDataConsumption = consumption_df.toPandas()

# COMMAND ----------

rawDataConsumption

# COMMAND ----------

def preprocess_inputs(estaciones, consumos, estaciones_no_utilizables):

    estaciones = estaciones.copy()
    consumos = consumos.copy()

    df = estaciones.merge(consumos,
                   how='inner',
                   on='IdComercializacion')
    
    df = df[~df['IdDispositivo'].isin(estaciones_no_utilizables)]

    df = df[['Nombre',
            'IdDispositivo',
            'IdComercializacion',
            'Tipo',
            'Fecha',
            'Volumen']]
    
    df.columns = ['estacion',
                  'iddispositivo',
                  'idcomercializacion',
                  'tipo',
                  'fecha',
                  'volumen']
    
    df = df.sort_values(by=['estacion','fecha'])
    
    return df


# COMMAND ----------

results = preprocess_inputs(estaciones=rawData,
                       consumos=rawDataConsumption,
                       estaciones_no_utilizables=estacionesNoUtilizables)
results.head()

# COMMAND ----------

schema = StructType([
    StructField("estacion", StringType(), True),
    StructField("iddispositivo", StringType(), True),
    StructField("idcomercializacion", StringType(), True),
    StructField("tipo", StringType(), True),
    StructField("fecha", DateType(), True),
    StructField("volumenm3", FloatType(), True)
    ])
df = spark.createDataFrame(results, schema = schema)

deltaTable = DeltaTable.forName(spark, 'analiticaefg.comercializacion.ingesta')

deltaTable.alias("t").merge(
    df.alias("s"),
    "t.estacion = s.estacion AND t.fecha = s.fecha"
).whenNotMatchedInsertAll().execute()

# COMMAND ----------


