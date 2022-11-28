# Databricks notebook source
import os
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
#from imblearn.pipeline import Pipeline as imbpipeline
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
from datetime import date


import sklearn
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

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
#is_training = dbutils.widgets.get("is_training") == "true"

# COMMAND ----------

today = date.today()
is_training = True

# COMMAND ----------

queryCT = 'SELECT * FROM ModeloAliadosBrilla.BaseCT'
queryGS = 'SELECT * FROM ModeloAliadosBrilla.BaseGS'
queryMotos = 'SELECT * FROM ModeloAliadosBrilla.BaseMotos'

# COMMAND ----------

print(f'Query para Aliados Canal Tradicional: {queryCT}')
print(f'Query para Aliados Grande Superficie: {queryGS}')
print(f'Query para Aliados Motos:             {queryMotos}')

# COMMAND ----------

dfCT = spark.read \
  .format("com.databricks.spark.sqldw") \
  .option("url", sqlDwUrl) \
  .option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
  .option("forwardSparkAzureStorageCredentials", "true") \
  .option("maxStrLength", "1024" ) \
  .option("query", queryCT) \
  .load()

rawDataCT = dfCT.toPandas()

dfGS = spark.read \
  .format("com.databricks.spark.sqldw") \
  .option("url", sqlDwUrl) \
  .option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
  .option("forwardSparkAzureStorageCredentials", "true") \
  .option("maxStrLength", "1024" ) \
  .option("query", queryGS) \
  .load()

rawDataGS = dfGS.toPandas()

dfMotos = spark.read \
  .format("com.databricks.spark.sqldw") \
  .option("url", sqlDwUrl) \
  .option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
  .option("forwardSparkAzureStorageCredentials", "true") \
  .option("maxStrLength", "1024" ) \
  .option("query", queryMotos) \
  .load()

rawDataMotos = dfMotos.toPandas()

# COMMAND ----------

rawDataCT = rawDataCT.drop(rawDataCT[rawDataCT['Aliado'] == 'CARDIF COLOMBIA SEGUROS GENERALES S.A.'].index,axis=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Preprocessing**

# COMMAND ----------

def preprocess_inputs(df):
    df = df.copy()
    
    df['Devolucion'] = df['Devoluciones']/df['NumeroDeVentas']

    # Se itera por las columnas Recency y Monetary, y crea bins divididos en quantiles.
    for cols in ['Recencia','Monetario','Devolucion','TicketPromedio','PlazoPromedio','Monetario6M','NumeroDeVentas']:
        # Dependiendo de en cual intervalo (bin) se encuentra, lo clasifica de 5 a 1.
        df[cols] = df[cols].astype('float')
        print(cols)
        if cols == 'Recencia':
            df[f'{cols}_bins'] = pd.cut(df[cols],5)
            sorted_bins = sorted(df[f'{cols}_bins'].value_counts().keys())
            r_score = []
            for j in df[cols]:
                counter = 5
                for v in sorted_bins:
                    if j in v:
                        break
                    else:
                        counter -= 1
                r_score.append(counter)

        elif cols == 'Devolucion':
            r_score = []
            for i in df[cols]:

              if i <= 0.05:
                r_score.append(0)
              else:
                r_score.append(1)

            # Dependiendo de en cual intervalo (bin) se encuentra, lo clasifica de 1 a 5.
        else:
            df[f'{cols}_bins'] = pd.qcut(df[cols],5,duplicates='drop')
            sorted_bins = sorted(df[f'{cols}_bins'].value_counts().keys())
            r_score = []
            for j in df[cols]:
                counter = 0
                for v in sorted_bins:
                    counter += 1
                    if j in v:
                        break
                r_score.append(counter)
                
        df[f'{cols}-Score'] = r_score
   

    df = df.drop(['Recencia_bins','Monetario_bins','NumeroDeVentas_bins','TicketPromedio_bins','Monetario6M_bins','PlazoPromedio_bins'],axis=1)
        
    return df

# COMMAND ----------

XCT = preprocess_inputs(rawDataCT)
XGS = preprocess_inputs(rawDataGS)
XMotos = preprocess_inputs(rawDataMotos)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Canal Tradicional**

# COMMAND ----------

col = ['Recencia-Score',
       'CARDIF',
      'Monetario-Score',
      'Devolucion-Score',
      'TicketPromedio-Score',
      'Monetario6M-Score']

colors = ['#DF2020', '#81DF20', '#2095DF','#F4D03F','#800080']

if is_training:
    kmeans = KMeans(n_clusters=5,random_state=9)
    XCT['cluster'] = kmeans.fit_predict(XCT[col])
    with open(f'/dbfs/FileStore/tables/KMeans_CT_{today}.pkl', 'wb') as handle:
        pkl.dump(model, handle, protocol = pkl.HIGHEST_PROTOCOL)
XCT['c'] = XCT.cluster.map({i:colors[i] for i in range(len(colors))})

pca = PCA(n_components=2)
new_X = pd.DataFrame(pca.fit_transform(XCT[col]),columns=['PC0','PC1'])
new_X['cluster'] = XCT['cluster']
new_X['c'] = XCT['c']

plt.figure(figsize=(15,10))
sns.scatterplot(new_X['PC0'],new_X['PC1'],hue=new_X['cluster'],palette=colors[:5])
plt.show()

# COMMAND ----------

XCT['TipoDeAliado'] = 0

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### **Grande Superficie**

# COMMAND ----------

col = ['CARDIF',
      'Recencia-Score',
      'Monetario-Score',
      'Devolucion-Score',
      'TicketPromedio-Score',
      'Monetario6M-Score']

n_clusters = 5

colors = ['#DF2020', '#81DF20', '#2095DF','#F4D03F','#800080']


kmeans = KMeans(n_clusters=n_clusters,random_state=9)
XGS['cluster'] = kmeans.fit_predict(XGS[col])
XGS['c'] = XGS.cluster.map({i:colors[i] for i in range(len(colors))})

pca = PCA(n_components=2)
new_X = pd.DataFrame(pca.fit_transform(XGS[col]),columns=['PC0','PC1'])
new_X['cluster'] = XGS['cluster']
new_X['c'] = XGS['c']

plt.figure(figsize=(15,10))
sns.scatterplot(new_X['PC0'],new_X['PC1'],hue=new_X['cluster'],palette=colors[:n_clusters])
plt.show()

# COMMAND ----------

XGS['TipoDeAliado'] = 2

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Motos**

# COMMAND ----------

col = ['Recencia-Score',
      'CARDIF',
      'Monetario-Score',
      'Devolucion-Score',
      'TicketPromedio-Score',
      'Monetario6M-Score']

colors = ['#DF2020', '#81DF20', '#2095DF','#F4D03F','#800080']


kmeans = KMeans(n_clusters=5,random_state=9)
XMotos['cluster'] = kmeans.fit_predict(XMotos[col])
XMotos['c'] = XMotos.cluster.map({i:colors[i] for i in range(len(colors))})

pca = PCA(n_components=2)
new_X = pd.DataFrame(pca.fit_transform(XMotos[col]),columns=['PC0','PC1'])
new_X['cluster'] = XMotos['cluster']
new_X['c'] = XMotos['c']

plt.figure(figsize=(15,10))
sns.scatterplot(new_X['PC0'],new_X['PC1'],hue=new_X['cluster'],palette=colors[:5])
plt.show()

# COMMAND ----------

XMotos['TipoDeAliado'] = 1

# COMMAND ----------

XCT[['IdPuntoVenta','cluster','TipoDeAliado']]

# COMMAND ----------

results = pd.concat([XCT[['IdPuntoVenta','cluster','TipoDeAliado']],XMotos[['IdPuntoVenta','cluster','TipoDeAliado']],XGS[['IdPuntoVenta','cluster','TipoDeAliado']]],axis=0).reset_index(drop=True)
results = results.rename(columns={'cluster':'Segmento'})

# COMMAND ----------

results

# COMMAND ----------

schema = StructType([
    StructField("IdPuntoVenta", IntegerType(), True),
    StructField("Segmento", IntegerType(), True),
    StructField("TipoDeAliado", IntegerType(), True)
  ])
df = spark.createDataFrame(results, schema = schema)
df.write \
.format("com.databricks.spark.sqldw") \
.option("url", sqlDwUrl) \
.option("forwardSparkAzureStorageCredentials", "true") \
.option("dbTable", "ModeloAliadosBrilla.SegmentosAliados") \
.option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
.mode("overwrite") \
.save()

# COMMAND ----------

groupbyCT = XCT.groupby('cluster').agg({'Recencia':['count','mean'],'CARDIF':['mean'],'NumeroDeVentas':['mean'],'Monetario':['mean'],'Devolucion':['mean'],'TicketPromedio':['mean'],'Monetario6M':['mean']})
groupbyGS = XGS.groupby('cluster').agg({'Recencia':['count','mean'],'CARDIF':['mean'],'NumeroDeVentas':['mean'],'Monetario':['mean'],'Devolucion':['mean'],'TicketPromedio':['mean'],'Monetario6M':['mean']})
groupbyMotos = XMotos.groupby('cluster').agg({'Recencia':['count','mean'],'CARDIF':['mean'],'NumeroDeVentas':['mean'],'Monetario':['mean'],'Devolucion':['mean'],'TicketPromedio':['mean'],'Monetario6M':['mean']})

# COMMAND ----------

clusterMotos = list(groupbyMotos.index)
countMotos = groupbyMotos['Recencia']['count']
recenciaMotos = groupbyMotos['Recencia']['mean']
cardifMotos = groupbyMotos['CARDIF']['mean']
numeroventasMotos = groupbyMotos['NumeroDeVentas']['mean']
monetarioMotos = groupbyMotos['Monetario']['mean']
devolucionMotos = groupbyMotos['Devolucion']['mean']
ticketMotos = groupbyMotos['TicketPromedio']['mean']
monetario6mMotos = groupbyMotos['Monetario6M']['mean']

# COMMAND ----------

clusterCT = list(groupbyCT.index)
countCT = groupbyCT['Recencia']['count']
recenciaCT = groupbyCT['Recencia']['mean']
cardifCT = groupbyCT['CARDIF']['mean']
numeroventasCT = groupbyCT['NumeroDeVentas']['mean']
monetarioCT = groupbyCT['Monetario']['mean']
devolucionCT = groupbyCT['Devolucion']['mean']
ticketCT = groupbyCT['TicketPromedio']['mean']
monetario6mCT = groupbyCT['Monetario6M']['mean']

# COMMAND ----------

clusterGS = list(groupbyGS.index)
countGS = groupbyGS['Recencia']['count']
recenciaGS = groupbyGS['Recencia']['mean']
cardifGS = groupbyGS['CARDIF']['mean']
numeroventasGS = groupbyGS['NumeroDeVentas']['mean']
monetarioGS = groupbyGS['Monetario']['mean']
devolucionGS = groupbyGS['Devolucion']['mean']
ticketGS = groupbyGS['TicketPromedio']['mean']
monetario6mGS = groupbyGS['Monetario6M']['mean']

# COMMAND ----------

dataPandas = pd.DataFrame({'Segmentos': clusterMotos,
                           'CantidadDeAliados': countMotos,
                           'Recencia': recenciaMotos,
                           'CARDIF': cardifMotos,
                           'NumeroDeVentas': numeroventasMotos,
                           'Monetario': monetarioMotos,
                           'Devoluciones': devolucionMotos,
                           'TicketPromedio': ticketMotos,
                           'Monetario6M': monetario6mMotos}).reset_index(drop=True)
schema = StructType([
    StructField("Segmentos", IntegerType(), True),
    StructField("CantidadDeAliados", IntegerType(), True),
    StructField("Recencia", FloatType(), True),
    StructField("CARDIF", FloatType(), True),
    StructField("NumeroDeVentas", FloatType(), True),
    StructField("Monetario", FloatType(), True),
    StructField("Devoluciones", FloatType(), True),
    StructField("TicketPromedio", FloatType(), True),
    StructField("Monetario6M", FloatType(), True),
    ])
df = spark.createDataFrame(dataPandas, schema = schema)
df.write \
.format("com.databricks.spark.sqldw") \
.option("url", sqlDwUrl) \
.option("forwardSparkAzureStorageCredentials", "true") \
.option("dbTable", "ModeloAliadosBrilla.ResumenMotos") \
.option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
.mode("overwrite") \
.save()

# COMMAND ----------

dataPandas = pd.DataFrame({'Segmentos': clusterCT,
                           'CantidadDeAliados': countCT,
                           'Recencia': recenciaCT,
                           'CARDIF': cardifCT,
                           'NumeroDeVentas': numeroventasCT,
                           'Monetario': monetarioCT,
                           'Devoluciones': devolucionCT,
                           'TicketPromedio': ticketCT,
                           'Monetario6M': monetario6mCT}).reset_index(drop=True)
schema = StructType([
    StructField("Segmentos", IntegerType(), True),
    StructField("CantidadDeAliados", IntegerType(), True),
    StructField("Recencia", FloatType(), True),
    StructField("CARDIF", FloatType(), True),
    StructField("NumeroDeVentas", FloatType(), True),
    StructField("Monetario", FloatType(), True),
    StructField("Devoluciones", FloatType(), True),
    StructField("TicketPromedio", FloatType(), True),
    StructField("Monetario6M", FloatType(), True),
    ])
df = spark.createDataFrame(dataPandas, schema = schema)
df.write \
.format("com.databricks.spark.sqldw") \
.option("url", sqlDwUrl) \
.option("forwardSparkAzureStorageCredentials", "true") \
.option("dbTable", "ModeloAliadosBrilla.ResumenCT") \
.option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
.mode("overwrite") \
.save()

# COMMAND ----------

dataPandas = pd.DataFrame({'Segmentos': clusterGS,
                           'CantidadDeAliados': countGS,
                           'Recencia': recenciaGS,
                           'CARDIF': cardifGS,
                           'NumeroDeVentas': numeroventasGS,
                           'Monetario': monetarioGS,
                           'Devoluciones': devolucionGS,
                           'TicketPromedio': ticketGS,
                           'Monetario6M': monetario6mGS}).reset_index(drop=True)
schema = StructType([
    StructField("Segmentos", IntegerType(), True),
    StructField("CantidadDeAliados", IntegerType(), True),
    StructField("Recencia", FloatType(), True),
    StructField("CARDIF", FloatType(), True),
    StructField("NumeroDeVentas", FloatType(), True),
    StructField("Monetario", FloatType(), True),
    StructField("Devoluciones", FloatType(), True),
    StructField("TicketPromedio", FloatType(), True),
    StructField("Monetario6M", FloatType(), True),
    ])
df = spark.createDataFrame(dataPandas, schema = schema)
df.write \
.format("com.databricks.spark.sqldw") \
.option("url", sqlDwUrl) \
.option("forwardSparkAzureStorageCredentials", "true") \
.option("dbTable", "ModeloAliadosBrilla.ResumenGS") \
.option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
.mode("overwrite") \
.save()

# COMMAND ----------


