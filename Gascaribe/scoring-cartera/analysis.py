# Databricks notebook source
import os
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, DateType
from datetime import date,datetime
today = datetime.now()
today_dt = today.strftime("%d-%m-%Y")

import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

# COMMAND ----------

dwDatabase = os.environ.get("DWH_NAME")
dwServer = os.environ.get("DWH_HOST")
dwUser = os.environ.get("DWH_USER")
dwPass = os.environ.get("DWH_PASS")
dwJdbcPort = os.environ.get("DWH_PORT")
dwJdbcExtraOptions = ""
sqlDwUrl = "jdbc:sqlserver://" + dwServer + ".database.windows.net:" + dwJdbcPort + ";database=" + dwDatabase + ";user=" + dwUser + ";password=" + dwPass + ";" + dwJdbcExtraOptions
storage_account_name = os.environ.get("BS_NAME")
blob_container = os.environ.get("BS_CONTAINER")
blob_storage = storage_account_name + ".blob.core.windows.net"
config_key = "fs.azure.account.key."+storage_account_name+".blob.core.windows.net"
blob_access_key = os.environ.get("BS_ACCESS_KEY")
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

A = 0.75

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

maxValue = df['Suspensiones'].max()
minValue = df['Suspensiones'].min()
varSuspensiones = []
for i in range(len(df['Suspensiones'])):
    if df['IdTipoProducto'][i] == 7055:
        varSuspensiones.append(None)
    else:
        varSuspensiones.append(normVariable(df['Suspensiones'][i],maxValue,minValue))

df['varSuspensiones'] = varSuspensiones

# COMMAND ----------

ponderado = []
for i in range(len(df)):
    if df['IdTipoProducto'][i] == 7014:
        xPago = 0.25
        xMora = 0.25
        xRefinanciaciones = 0.25
        xSuspensiones = 0.25

        ponderado.append(xPago*df['VarPago'][i] + xMora*df['VarMora'][i] + xRefinanciaciones*df['VarRefinanciaciones'][i] + xSuspensiones*df['varSuspensiones'][i])
    else:
        xPago = 0.34
        xMora = 0.33
        xRefinanciaciones = 0.33

        ponderado.append(xPago*df['VarPago'][i] + xMora*df['VarMora'][i] + xRefinanciaciones*df['VarRefinanciaciones'][i])
        
df['Ponderado'] = ponderado

# COMMAND ----------

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### **Writing Into DWH**

# COMMAND ----------

results = df[['IdTipoProducto','IdProducto','Intervalo','VarPago','VarMora','VarRefinanciaciones','varSuspensiones','Ponderado']]
results['FechaPrediccion'] = today_dt
results['FechaPrediccion'] = pd.to_datetime(results['FechaPrediccion'])
#results['Valido'] = 1

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
    StructField("Ponderado", FloatType(), True),
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


