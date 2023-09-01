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

def processing_inputs(df):
    df = df.copy()

    # Borrar columna Red de la tabla
    df = df.drop('Red',axis=1)

    # Renombrar columnas
    df.columns = ['iddispositivo','estacion','tipo','fecha','volumenm3']
    # Reordenar columnas
    df = df[['estacion','iddispositivo','tipo','fecha','volumenm3']]

    # Reemplazar valores 0E-8 por 0
    df['volumenm3'] = df['volumenm3'].apply(lambda x: 0 if x == 0E-8 else x)

    df['volumenm3'] = df['volumenm3'].astype('float')

    return df

# COMMAND ----------

X = processing_inputs(rawData)

# COMMAND ----------

schema = StructType([
    StructField("estacion", StringType(), True),
    StructField("iddispositivo", StringType(), True),
    StructField("tipo", StringType(), True),
    StructField("fecha", DateType(), True),
    StructField("volumenm3", FloatType(), True)
    ])
df = spark.createDataFrame(X, schema = schema)

deltaTable = DeltaTable.forName(spark, 'analiticagdc.comercializacion.ingesta')

deltaTable.alias("t").merge(
    df.alias("s"),
    "t.estacion = s.estacion AND t.fecha = s.fecha AND t.volumenm3 = s.volumenm3"
).whenNotMatchedInsertAll().execute()
