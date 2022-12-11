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

import sklearn
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

# COMMAND ----------

dwDatabase = os.environ.get("DWH_NAME_GDC")
dwServer = os.environ.get("DWH_HOST_GDC")
dwUser = os.environ.get("DWH_USER_GDC")
dwPass = os.environ.get("DWH_PASS_GDC")
dwJdbcPort = os.environ.get("DWH_PORT_GDC")
dwJdbcExtraOptions = ""
sqlDwUrl = "jdbc:sqlserver://" + dwServer + ".database.windows.net:" + dwJdbcPort + ";database=" + dwDatabase + ";user=" + dwUser + ";password=" + dwPass + ";" + dwJdbcExtraOptions
storage_account_name = os.environ.get("BS_NAME_GDC")
blob_container = os.environ.get("BS_CONTAINER_GDC")
blob_storage = storage_account_name + ".blob.core.windows.net"
config_key = "fs.azure.account.key."+storage_account_name+".blob.core.windows.net"
blob_access_key = os.environ.get("BS_ACCESS_KEY_GDC")
spark.conf.set(config_key, blob_access_key)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC ### **Data Collection**

# COMMAND ----------

queryCT = 'SELECT * FROM ModeloAliadosBrilla.BaseCT'
queryGS = 'SELECT * FROM ModeloAliadosBrilla.BaseGS'

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

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC ### **Preprocessing**

# COMMAND ----------

def preprocess_inputs(df):
    df = df.copy()

    df['Devolucion'] = df['Devoluciones']/df['NumeroDeVentas']
    df['Devolucion'] = df['Devolucion']
    
    
    # Se itera por las columnas Recency y Monetary, y crea bins divididos en quantiles.
    for cols in ['Recencia','NumeroDeVentas','Monetario','Devolucion','TicketPromedio','PlazoPromedio','Monetario6M']:
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

    freq_score = []


    df = df.drop(['Recencia_bins',
                  'Monetario_bins',
                  'NumeroDeVentas_bins',
                  'Devolucion_bins',
                  'TicketPromedio_bins',
                  'PlazoPromedio_bins',
                  'Monetario6M_bins'],axis=1)
        
    return df

# COMMAND ----------

XCT = preprocess_inputs(rawDataCT)
XGS = preprocess_inputs(rawDataGS)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC ### **Canal Tradicional**

# COMMAND ----------

with open(f'/dbfs/ModelosGDC/Aliados/KMeans_CT.pkl', 'rb') as handle:
    model = pkl.load(handle)

col = ['Recencia-Score',
      'NumeroDeVentas-Score',
      'Monetario-Score',
      'Devolucion-Score',
      'TicketPromedio-Score',
      'Monetario6M-Score']
    
XCT['cluster'] = model.predict(XCT[col])

centroidsCT = model.cluster_centers_
clustersCT = pd.DataFrame(centroidsCT,columns=col)
clustersCT['cluster'] = model.predict(clustersCT[col]) 
clustersCT['magnitude'] = np.sqrt(((clustersCT['Recencia-Score']**2) + (clustersCT['Monetario-Score']**2)+ (clustersCT['Devolucion-Score']**2)+ (clustersCT['TicketPromedio-Score']**2)+ (clustersCT['Monetario6M-Score']**2)))

clustersCT['name'] = [0,0,0,0]
clustersCT['name'].iloc[clustersCT['Monetario-Score'].idxmax()] = 'Premium'
clustersCT['name'].iloc[clustersCT['Monetario-Score'].idxmin()] = 'Estandar'
clustersCT['name'].iloc[clustersCT['Monetario-Score'] == list(clustersCT['Monetario-Score'].nlargest(2))[1]] = 'Elite'
clustersCT['name'].iloc[clustersCT['Monetario-Score'] == list(clustersCT['Monetario-Score'].nsmallest(2))[1]] = 'Plus'

XCTMerged = XCT.merge(clustersCT[['cluster','name']], on='cluster',how='left')

XCTMerged['TipoDeAliado'] = 0

XCTMerged = XCTMerged.rename(columns={'IdContratista':'Id'})


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC ### **Grande Superficie**

# COMMAND ----------

with open(f'/dbfs/ModelosGDC/Aliados/KMeans_GS.pkl', 'rb') as handle:
    model = pkl.load(handle)

col = ['Recencia-Score',
      'NumeroDeVentas-Score',
      'Monetario-Score',
      'Devolucion-Score',
      'TicketPromedio-Score',
      'Monetario6M-Score']
    
XGS['cluster'] = model.predict(XGS[col])

centroidsGS = model.cluster_centers_
clustersGS = pd.DataFrame(centroidsGS,columns=col)
clustersGS['cluster'] = model.predict(clustersGS[col]) 
clustersGS['magnitude'] = np.sqrt(((clustersGS['Recencia-Score']**2) + (clustersGS['Monetario-Score']**2)+ (clustersGS['Devolucion-Score']**2)+ (clustersGS['TicketPromedio-Score']**2)+ (clustersGS['Monetario6M-Score']**2)))

clustersGS['name'] = [0,0,0]
clustersGS['name'].iloc[clustersGS['Monetario-Score'].idxmax()] = 'Premium'
clustersGS['name'].iloc[clustersGS['Monetario-Score'].idxmin()] = 'Estandar'
clustersGS['name'].iloc[clustersGS['Monetario-Score'] == list(clustersGS['Monetario-Score'].nlargest(2))[1]] = 'Elite'

XGSMerged = XGS.merge(clustersGS[['cluster','name']], on='cluster',how='left')

XGSMerged['TipoDeAliado'] = 1

XGSMerged = XGSMerged.rename(columns={'IdPuntoVenta':'Id'})


# COMMAND ----------

results = pd.concat([XCTMerged[['Id','cluster','name','TipoDeAliado']],XGSMerged[['Id','cluster','name','TipoDeAliado']]],axis=0).reset_index(drop=True)
results = results.rename(columns={'cluster':'Segmento',
                                  'name': 'NombreSegmento'})

results['FechaPrediccion'] = today_dt
results['FechaPrediccion'] = pd.to_datetime(results['FechaPrediccion'])

# COMMAND ----------

schema = StructType([
    StructField("Id", IntegerType(), True),
    StructField("Segmento", IntegerType(), True),
    StructField("NombreSegmento", StringType(), True),
    StructField("TipoDeAliado", IntegerType(), True),
    StructField("FechaPrediccion", DateType(), True)
  ])
df = spark.createDataFrame(results, schema = schema)
df.write \
.format("com.databricks.spark.sqldw") \
.option("url", sqlDwUrl) \
.option("forwardSparkAzureStorageCredentials", "true") \
.option("dbTable", "ModeloAliadosBrilla.StageSegmentosAliados") \
.option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
.mode("overwrite") \
.save()

# COMMAND ----------

groupbyCT = XCTMerged.groupby('name').agg({'Recencia':['count','mean'],'NumeroDeVentas':['mean'],'Monetario':['mean'],'Devolucion':['mean'],'TicketPromedio':['mean'],'Monetario6M':['mean']})
groupbyGS = XGSMerged.groupby('name').agg({'Recencia':['count','mean'],'NumeroDeVentas':['mean'],'Monetario':['mean'],'Devolucion':['mean'],'TicketPromedio':['mean'],'Monetario6M':['mean']})

# COMMAND ----------

clusterCT = list(groupbyCT.index)
countCT = groupbyCT['Recencia']['count']
recenciaCT = groupbyCT['Recencia']['mean']
numeroventasCT = groupbyCT['NumeroDeVentas']['mean']
monetarioCT = groupbyCT['Monetario']['mean']
devolucionCT = groupbyCT['Devolucion']['mean']
ticketCT = groupbyCT['TicketPromedio']['mean']
monetario6mCT = groupbyCT['Monetario6M']['mean']

clusterGS = list(groupbyGS.index)
countGS = groupbyGS['Recencia']['count']
recenciaGS = groupbyGS['Recencia']['mean']
numeroventasGS = groupbyGS['NumeroDeVentas']['mean']
monetarioGS = groupbyGS['Monetario']['mean']
devolucionGS = groupbyGS['Devolucion']['mean']
ticketGS = groupbyGS['TicketPromedio']['mean']
monetario6mGS = groupbyGS['Monetario6M']['mean']

# COMMAND ----------

dataPandas = pd.DataFrame({'Segmentos': clusterCT,
                           'CantidadDeAliados': countCT,
                           'Recencia': recenciaCT,
                           'NumeroDeVentas': numeroventasCT,
                           'Monetario': monetarioCT,
                           'Devoluciones': devolucionCT,
                           'TicketPromedio': ticketCT,
                           'Monetario6M': monetario6mCT}).reset_index(drop=True)
schema = StructType([
    StructField("Segmentos", StringType(), True),
    StructField("CantidadDeAliados", IntegerType(), True),
    StructField("Recencia", FloatType(), True),
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
                           'NumeroDeVentas': numeroventasGS,
                           'Monetario': monetarioGS,
                           'Devoluciones': devolucionGS,
                           'TicketPromedio': ticketGS,
                           'Monetario6M': monetario6mGS}).reset_index(drop=True)
schema = StructType([
    StructField("Segmentos", StringType(), True),
    StructField("CantidadDeAliados", IntegerType(), True),
    StructField("Recencia", FloatType(), True),
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


