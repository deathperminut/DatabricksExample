# Databricks notebook source
import os
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, DateType
from delta.tables import *
from pyspark.sql.functions import *
from datetime import date,datetime
today = datetime.now()
today_dt = today.strftime("%d-%m-%Y")

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from scipy.spatial.distance import cdist

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

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ### **Queries**

# COMMAND ----------

query = "SELECT * FROM ScoringCartera.FactScoringResidencial"

# COMMAND ----------

dfSpark = spark.read \
  .format("com.databricks.spark.sqldw") \
  .option("url", sqlDwUrl) \
  .option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
  .option("forwardSparkAzureStorageCredentials", "true") \
  .option("maxStrLength", "1024" ) \
  .option("query", query) \
  .load()

df = dfSpark.toPandas()

# COMMAND ----------

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### **Remove inactive products**

# COMMAND ----------

df = df[df['Facturacion'] != 0].reset_index(drop=True)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ### **Processing**

# COMMAND ----------

def normVariable(value,maxValue,minValue):
    
    normValue = (float(value) - minValue)/(maxValue - minValue)
    
    return normValue

# COMMAND ----------

quantileValue = df['Pagos'].quantile(0.95)
minValue = df['Pagos'].min()

df['InP'] = df['Pagos'].apply(lambda x: 1 if x > quantileValue else normVariable(x,quantileValue,minValue))

# COMMAND ----------

A = 1

df['VarPago'] = A*(df['Pagos']/df['Facturacion']) + (1 - A)*df['InP']
df['VarPago'] = df['VarPago'].apply(lambda x: 1 if x > 1 else x)
df['VarPago'] = df['VarPago'].apply(lambda x: 0 if x < 0 else x)
df['VarPago'] = df['VarPago'].fillna(0)

# COMMAND ----------

df['VarMora'] = 1 - (0.3*df['E30'] + 0.5*df['E60'] + 0.8*df['E90'] + df['EM90'])/df['Intervalo']

# COMMAND ----------

maxValue = df['Refinanciaciones'].max()
minValue = df['Refinanciaciones'].min()
df['VarRefinanciaciones'] = df['Refinanciaciones'].apply(lambda x: 1 - normVariable(x,maxValue,minValue))

# COMMAND ----------

suspensiones = {
    6:183,
    12:366,
    24:731
}

for i in range(len(df)):
    for intervalo,suspensionMax in suspensiones.items():
        if df['Intervalo'][i] == intervalo and df['DiasSuspendidos'][i] > suspensionMax:
            df['DiasSuspendidos'][i] = suspensionMax
        else:
            pass

# COMMAND ----------

df6 = df[df['Intervalo'] == 6].reset_index(drop=True)
df12 = df[df['Intervalo'] == 12].reset_index(drop=True)
df24 = df[df['Intervalo'] == 24].reset_index(drop=True)

fulldfs = [df6,df12,df24]

for dfi in fulldfs:
    maxValue = dfi['DiasSuspendidos'].max()
    minValue = dfi['DiasSuspendidos'].min()
    varSuspensiones = []
    for i in range(len(dfi['DiasSuspendidos'])):
        if dfi['IdTipoProducto'][i] == 7055:
            varSuspensiones.append(None)
        else:
            varSuspensiones.append(1 - normVariable(dfi['DiasSuspendidos'][i],maxValue,minValue))

    dfi['varSuspensiones'] = varSuspensiones
    
    maxCastigo = dfi['ConteoCastigado'].max()
    varCastigo = []
    for i in range(len(dfi['ConteoCastigado'])):
        varCastigo.append(dfi['ConteoCastigado'][i]/maxCastigo)
    
    dfi['varCastigo'] = varCastigo
        
    
newdf = pd.concat([df6,df12,df24],axis=0).reset_index(drop=True)

# COMMAND ----------

ponderado = []
for i in range(len(newdf)):
    if newdf['IdTipoProducto'][i] == 7014:
        xPago = 0.15
        xMora = 0.35
        xRefinanciaciones = 0.25
        xSuspensiones = 0.25

        score = (xPago*newdf['VarPago'][i] + xMora*newdf['VarMora'][i] + xRefinanciaciones*newdf['VarRefinanciaciones'][i] + xSuspensiones*newdf['varSuspensiones'][i])*(1 - newdf['varCastigo'][i])
        
        if score < 0:
            score = 0
        else:
            pass

        ponderado.append(score)
        
    else:
        xPago = 0.14
        xMora = 0.53
        xRefinanciaciones = 0.33

        score = (xPago*newdf['VarPago'][i] + xMora*newdf['VarMora'][i] + xRefinanciaciones*newdf['VarRefinanciaciones'][i])*(1 - newdf['varCastigo'][i])
        
        if score < 0:
            score = 0
        else:
            pass

        ponderado.append(score)
        
        
newdf['Ponderado'] = ponderado

# COMMAND ----------

df = newdf.copy()

# COMMAND ----------

cluster = []
for i in range(len(df)):
    if df['Ponderado'][i] <= 0.55 or df['Castigado'][i] == 1:
        cluster.append(0)
    elif df['Ponderado'][i] > 0.55 and df['Ponderado'][i] <= 0.7:
        cluster.append(1)
    elif df['Ponderado'][i] > 0.7 and df['Ponderado'][i] <= 0.85:
        cluster.append(2)
    elif df['Ponderado'][i] > 0.85 and df['Ponderado'][i] < 1:
        cluster.append(3)
    elif df['Ponderado'][i] == 1:
        cluster.append(4)

df['Segmento'] = cluster

# COMMAND ----------

df['SegmentoNombre'] = df['Segmento'].replace({
    0:'Pesimo',
    1:'Malo',
    2:'Regular',
    3:'Bueno',
    4:'Excelente'
})

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### **Writing Into DWH**

# COMMAND ----------

df.head()

# COMMAND ----------

results = df[['IdTipoProducto','IdProducto','Intervalo','VarPago','VarMora','VarRefinanciaciones','varSuspensiones','DiasSuspendidos','Castigado','ConteoCastigado','Ponderado','Segmento','SegmentoNombre']]
results['FechaPrediccion'] = today_dt
results['FechaPrediccion'] = pd.to_datetime(results['FechaPrediccion']).dt.strftime('%Y-%m-%d')

results = results.rename(columns={'VarPago':'PagosFacturacion',
                                 'VarMora':'MorasEscaladas',
                                 'VarRefinanciaciones':'RefinanciacionesEscaladas',
                                 'VarSuspensiones':'SuspensionesEscaladas'})

# COMMAND ----------

results.head()

# COMMAND ----------

schema = StructType([
    StructField("IdTipoProducto", IntegerType(), True),
    StructField("IdProducto", IntegerType(), True),
    StructField("Intervalo", IntegerType(), True),  
    StructField("PagosFacturacion", FloatType(), True),
    StructField("MorasEscaladas", FloatType(), True),
    StructField("RefinanciacionesEscaladas", FloatType(), True),
    StructField("SuspensionesEscaladas", FloatType(), True),
    StructField("DiasSuspendidos", IntegerType(), True),
    StructField("Castigado", IntegerType(), True),
    StructField("ConteoCastigado", IntegerType(), True),
    StructField("Ponderado", FloatType(), True),
    StructField("Segmento", IntegerType(), True),
    StructField("SegmentoNombre", StringType(), True),
    StructField("FechaPrediccion", DateType(), True)
    ])

df = spark.createDataFrame(results, schema = schema)

df.write \
.format("com.databricks.spark.sqldw") \
.option("url", sqlDwUrl) \
.option("forwardSparkAzureStorageCredentials", "true") \
.option("dbTable", "ScoringCartera.FactScoring") \
.option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
.mode("overwrite") \
.save()

# COMMAND ----------

df.write.mode('overwrite').saveAsTable('analiticagdc.scoringcartera.factscoring')

# COMMAND ----------

df.write.mode('append').saveAsTable('analiticagdc.scoringcartera.factscoring_historia')
