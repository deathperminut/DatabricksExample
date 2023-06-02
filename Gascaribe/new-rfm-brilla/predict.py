# Databricks notebook source
import os
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import msal
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, DateType
from datetime import date,datetime
today = datetime.now()
today_dt = today.strftime("%d-%m-%Y")

from pyspark.sql.functions import lit

import sklearn
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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
# MAGIC ### **Data Collection**

# COMMAND ----------

query = 'SELECT * FROM ModeloRFMBrilla.BaseRFM'

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

rawData.head()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ### **Preprocessing**

# COMMAND ----------

def preprocess_inputs(df):
    
    df = df.copy()
    df = df[df['Recency'] <= 49]
   
    # Se itera por las columnas Recency y Monetary, y crea bins divididos en quantiles.
    for cols in ['Recency','Monetary']:
        df[f'{cols}_bins'] = pd.qcut(df[cols],5)
        sorted_bins = sorted(df[f'{cols}_bins'].value_counts().keys())
        # Dependiendo de en cual intervalo (bin) se encuentra, lo clasifica de 5 a 1.
        if cols == 'Recency':
            r_score = []
            for j in df[cols]:
                counter = 5
                for v in sorted_bins:
                    if j in v:
                        break
                    else:
                        counter -= 1
                r_score.append(counter)
            # Dependiendo de en cual intervalo (bin) se encuentra, lo clasifica de 1 a 5.
            
        else:
            r_score = []
            for j in df[cols]:
                counter = 0
                for v in sorted_bins:
                    counter += 1
                    if j in v:
                        break
                r_score.append(counter)
                
        df[f'{cols}-Score'] = r_score
        
    # Esta clasificacion es manual, y se clasifica de 1 a 3.
    freq_score = []
    for i in df['Frequency']:
        if i in [1,2,3]:
            freq_score.append(i)
        elif i in [4,5]:
            freq_score.append(4)
        elif i >= 6:
            freq_score.append(5)
        
    df['Frequency-Score'] = freq_score

    df = df.drop(['Recency_bins','Monetary_bins'],axis=1)
    
    print(f'Numero de Usuarios Activos: {len(df)}')
        
    return df

# COMMAND ----------

X = preprocess_inputs(rawData)
X.head()

# COMMAND ----------

def get_inactivos(df):
    inactivos = df[df['Recency'] > 49]
    
    inactivos_df = pd.DataFrame({'Identificacion':inactivos['Identificacion'],
                                 'Recency':inactivos['Recency'],
                                 'Frequency':inactivos['Frequency'],
                                 'Monetary':inactivos['Monetary'],
                                'cluster':len(inactivos)*[4],
                                'name':len(inactivos)*['Inactivos']})
    print(f'Numero de Usuarios Inactivos: {len(inactivos_df)}')
    return inactivos_df

# COMMAND ----------

inactivos_df = get_inactivos(rawData)
inactivos_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ### **Prediction**

# COMMAND ----------

with open(f'/dbfs/ModelosGDC/RFMBrilla/KMeans_RFM.pkl', 'rb') as handle:
    model = pkl.load(handle)
    
X['cluster'] = model.predict(X[['Recency-Score','Monetary-Score','Frequency-Score']])

centroids = model.cluster_centers_
clusters = pd.DataFrame(centroids, columns=['Recency-Score','Monetary-Score','Frequency-Score'])

clusters['cluster'] = model.predict(clusters[['Recency-Score','Monetary-Score','Frequency-Score']]) 
clusters['magnitude'] = np.sqrt(((clusters['Recency-Score']**2) + (clusters['Monetary-Score']**2) + (clusters['Frequency-Score']**2)))
clusters['name'] = [0,0,0,0]
clusters['name'].iloc[clusters['magnitude'].idxmax()] = 'Diamante'
clusters['name'].iloc[clusters['magnitude'].idxmin()] = 'Bronce'

for i in range(len(clusters)):
    if (clusters['Recency-Score'].iloc[i] > 3.5) and (clusters['Frequency-Score'].iloc[i] < 2.3) and (clusters['magnitude'].iloc[i] != clusters['magnitude'].max()) and (clusters['magnitude'].iloc[i] != clusters['magnitude'].min()):
        clusters['name'].iloc[i] = 'Nuevo'
    elif (clusters['Recency-Score'].iloc[i] < 3.5) and (clusters['Frequency-Score'].iloc[i] > 2.3) and (clusters['magnitude'].iloc[i] != clusters['magnitude'].max()) and (clusters['magnitude'].iloc[i] != clusters['magnitude'].min()):
        clusters['name'].iloc[i] = 'Plata'
    else:
        pass
      
XMerged = X.merge(clusters[['cluster','name']],on='cluster',how='left')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ### **Writing Data Into DWH**

# COMMAND ----------

newX = pd.concat([XMerged[['Identificacion', 'Recency', 'Frequency', 'Monetary', 'cluster','name']],inactivos_df[['Identificacion', 'Recency', 'Frequency', 'Monetary', 'cluster','name']]],axis=0).reset_index(drop=True)
newX.head()

# COMMAND ----------

results = newX[['Identificacion','cluster','name']]
results['FechaPrediccion'] = today_dt
results['FechaPrediccion'] = pd.to_datetime(results['FechaPrediccion'])
#results['Valido'] = 1

results = results.rename(columns={'cluster':'Segmento',
                                 'name':'NombreSegmento'})

# COMMAND ----------

schema = StructType([
    StructField("Identificacion", StringType(), True),
    StructField("Segmento", IntegerType(), True),
    StructField("NombreSegmento", StringType(), True),
    StructField("FechaPrediccion", DateType(), True)
    ])
df = spark.createDataFrame(results, schema = schema)
df.write \
.format("com.databricks.spark.sqldw") \
.option("url", sqlDwUrl) \
.option("forwardSparkAzureStorageCredentials", "true") \
.option("dbTable", "ModeloRFMBrilla.StageSegmentosRFM") \
.option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
.mode("overwrite") \
.save()

# COMMAND ----------

df = df.withColumn("is_current", lit(1))

# COMMAND ----------

df.write.mode('overwrite').saveAsTable('analiticagdc.brilla.segmentosrfm')

# COMMAND ----------

groupbyUsuarios = newX.groupby('name').agg({'Recency':['count','mean'],'Frequency':['mean'],'Monetary':['mean']})

clusterUsuarios = list(groupbyUsuarios.index)
countUsuarios = groupbyUsuarios['Recency']['count']
recencyUsuarios = groupbyUsuarios['Recency']['mean']
frequencyUsuarios = groupbyUsuarios['Frequency']['mean']
monetaryUsuarios = groupbyUsuarios['Monetary']['mean']

# COMMAND ----------

dataPandas = pd.DataFrame({'Segmento': clusterUsuarios,
                           'CantidadDeUsuarios': countUsuarios,
                           'Recencia': recencyUsuarios,
                           'Frecuencia': frequencyUsuarios,
                           'Monetario': monetaryUsuarios}).reset_index(drop=True)
schema = StructType([
    StructField("Segmento", StringType(), True),
    StructField("CantidadDeUsuarios", IntegerType(), True),
    StructField("Recencia", FloatType(), True),
    StructField("Frecuencia", FloatType(), True),
    StructField("Monetario", FloatType(), True)
    ])
df = spark.createDataFrame(dataPandas, schema = schema)
df.write \
.format("com.databricks.spark.sqldw") \
.option("url", sqlDwUrl) \
.option("forwardSparkAzureStorageCredentials", "true") \
.option("dbTable", "ModeloRFMBrilla.ResumenSegmentos") \
.option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
.mode("overwrite") \
.save()
