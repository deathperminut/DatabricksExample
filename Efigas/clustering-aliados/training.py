# Databricks notebook source
import os
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
#from imblearn.pipeline import Pipeline as imbpipeline
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, DateType
from datetime import date,datetime
today = datetime.now()
today_dt = today.strftime("%d-%m-%Y")

import sklearn
from sklearn import metrics
from sklearn.cluster import KMeans
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

queryCT = 'SELECT * FROM ModeloAliadosBrilla.BaseCT'
queryGS = 'SELECT * FROM ModeloAliadosBrilla.BaseGS'
queryMotos = 'SELECT * FROM ModeloAliadosBrilla.BaseMotos'

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

rawDataCT = rawDataCT.drop(rawDataCT[rawDataCT['Aliado'] == 'CARDIF COLOMBIA SEGUROS GENERALES S.A.'].index,axis=0)

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


kmeans = KMeans(n_clusters=5,random_state=9)
XCT['cluster'] = kmeans.fit_predict(XCT[col])
XCT['c'] = XCT.cluster.map({i:colors[i] for i in range(len(colors))})

with open(f'/dbfs/ModelosEFG/Aliados/KMeans_CT.pkl', 'wb') as handle:
    pkl.dump(kmeans, handle, protocol = pkl.HIGHEST_PROTOCOL)


pca = PCA(n_components=2)
new_X = pd.DataFrame(pca.fit_transform(XCT[col]),columns=['PC0','PC1'])
new_X['cluster'] = XCT['cluster']
new_X['c'] = XCT['c']

plt.figure(figsize=(15,10))
sns.scatterplot(new_X['PC0'],new_X['PC1'],hue=new_X['cluster'],palette=colors[:5])
plt.show()

# COMMAND ----------

centroidsCT = kmeans.cluster_centers_
clustersCT = pd.DataFrame(centroidsCT,columns=col)
clustersCT['cluster'] = kmeans.predict(clustersCT[col]) 
clustersCT['magnitude'] = np.sqrt(((clustersCT['Recencia-Score']**2) + (clustersCT['CARDIF']**2) + (clustersCT['Monetario-Score']**2)+ (clustersCT['Devolucion-Score']**2)+ (clustersCT['TicketPromedio-Score']**2)+ (clustersCT['Monetario6M-Score']**2)))

clustersCT['name'] = [0,0,0,0,0]
clustersCT['name'].iloc[clustersCT['Monetario-Score'].idxmax()] = 'Diamante'
clustersCT['name'].iloc[clustersCT['Monetario-Score'].idxmin()] = 'Bronce'
clustersCT['name'].iloc[clustersCT['Monetario-Score'] == list(clustersCT['Monetario-Score'].nlargest(2))[1]] = 'Platino'
clustersCT['name'].iloc[clustersCT['Monetario-Score'] == list(clustersCT['Monetario-Score'].nsmallest(2))[1]] = 'Plata'
clustersCT['name'].iloc[clustersCT['name'] == 0] = 'Oro'

XCTMerged = XCT.merge(clustersCT[['cluster','name']], on='cluster',how='left')

XCTMerged['TipoDeAliado'] = 0

XCTMerged = XCTMerged.rename(columns={'IdContratista':'Id'})

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

with open(f'/dbfs/ModelosEFG/Aliados/KMeans_GS.pkl', 'wb') as handle:
    pkl.dump(kmeans, handle, protocol = pkl.HIGHEST_PROTOCOL)

pca = PCA(n_components=2)
new_X = pd.DataFrame(pca.fit_transform(XGS[col]),columns=['PC0','PC1'])
new_X['cluster'] = XGS['cluster']
new_X['c'] = XGS['c']

plt.figure(figsize=(15,10))
sns.scatterplot(new_X['PC0'],new_X['PC1'],hue=new_X['cluster'],palette=colors[:n_clusters])
plt.show()

# COMMAND ----------

centroidsGS = kmeans.cluster_centers_
clustersGS = pd.DataFrame(centroidsGS,columns=col)
clustersGS['cluster'] = kmeans.predict(clustersGS[col]) 
clustersGS['magnitude'] = np.sqrt(((clustersGS['Recencia-Score']**2) + (clustersGS['CARDIF']**2) + (clustersGS['Monetario-Score']**2)+ (clustersGS['Devolucion-Score']**2)+ (clustersGS['TicketPromedio-Score']**2)+ (clustersGS['Monetario6M-Score']**2)))

clustersGS['name'] = [0,0,0,0,0]
clustersGS['name'].iloc[clustersGS['Monetario-Score'].idxmax()] = 'Diamante'
clustersGS['name'].iloc[clustersGS['Monetario-Score'].idxmin()] = 'Bronce'
clustersGS['name'].iloc[clustersGS['Monetario-Score'] == list(clustersGS['Monetario-Score'].nlargest(2))[1]] = 'Platino'
clustersGS['name'].iloc[clustersGS['Monetario-Score'] == list(clustersGS['Monetario-Score'].nsmallest(2))[1]] = 'Plata'
clustersGS['name'].iloc[clustersGS['name'] == 0] = 'Oro'

XGSMerged = XGS.merge(clustersGS[['cluster','name']], on='cluster',how='left')

XGSMerged['TipoDeAliado'] = 2

XGSMerged = XGSMerged.rename(columns={'IdPuntoVenta':'Id'})

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

with open(f'/dbfs/ModelosEFG/Aliados/KMeans_Motos.pkl', 'wb') as handle:
    pkl.dump(kmeans, handle, protocol = pkl.HIGHEST_PROTOCOL)

pca = PCA(n_components=2)
new_X = pd.DataFrame(pca.fit_transform(XMotos[col]),columns=['PC0','PC1'])
new_X['cluster'] = XMotos['cluster']
new_X['c'] = XMotos['c']

plt.figure(figsize=(15,10))
sns.scatterplot(new_X['PC0'],new_X['PC1'],hue=new_X['cluster'],palette=colors[:5])
plt.show()

# COMMAND ----------

centroidsMotos = kmeans.cluster_centers_
clustersMotos = pd.DataFrame(centroidsMotos,columns=col)
clustersMotos['cluster'] = kmeans.predict(clustersMotos[col]) 
clustersMotos['magnitude'] = np.sqrt(((clustersMotos['Recencia-Score']**2) + (clustersMotos['CARDIF']**2) + (clustersMotos['Monetario-Score']**2)+ (clustersMotos['Devolucion-Score']**2)+ (clustersMotos['TicketPromedio-Score']**2)+ (clustersMotos['Monetario6M-Score']**2)))

clustersMotos['name'] = [0,0,0,0,0]
clustersMotos['name'].iloc[clustersMotos['Monetario-Score'].idxmax()] = 'Diamante'
clustersMotos['name'].iloc[clustersMotos['Monetario-Score'].idxmin()] = 'Bronce'
clustersMotos['name'].iloc[clustersMotos['Monetario-Score'] == list(clustersMotos['Monetario-Score'].nlargest(2))[1]] = 'Platino'
clustersMotos['name'].iloc[clustersMotos['Monetario-Score'] == list(clustersMotos['Monetario-Score'].nsmallest(2))[1]] = 'Plata'
clustersMotos['name'].iloc[clustersMotos['name'] == 0] = 'Oro'

XMotosMerged = XMotos.merge(clustersMotos[['cluster','name']], on='cluster',how='left')

XMotosMerged['TipoDeAliado'] = 1

XMotosMerged = XMotosMerged.rename(columns={'IdContratista':'Id'})

# COMMAND ----------

results = pd.concat([XCTMerged[['Id','cluster','name','TipoDeAliado']],XMotosMerged[['Id','cluster','name','TipoDeAliado']],XGSMerged[['Id','cluster','name','TipoDeAliado']]],axis=0).reset_index(drop=True)
results = results.rename(columns={'cluster':'Segmento',
                                  'name': 'NombreSegmento'})

results['FechaPrediccion'] = today_dt
results['FechaPrediccion'] = pd.to_datetime(results['FechaPrediccion'])

# COMMAND ----------

results

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

groupbyCT = XCTMerged.groupby('name').agg({'Recencia':['count','mean'],'CARDIF':['mean'],'NumeroDeVentas':['mean'],'Monetario':['mean'],'Devolucion':['mean'],'TicketPromedio':['mean'],'Monetario6M':['mean']})
groupbyGS = XGSMerged.groupby('name').agg({'Recencia':['count','mean'],'CARDIF':['mean'],'NumeroDeVentas':['mean'],'Monetario':['mean'],'Devolucion':['mean'],'TicketPromedio':['mean'],'Monetario6M':['mean']})
groupbyMotos = XMotosMerged.groupby('name').agg({'Recencia':['count','mean'],'CARDIF':['mean'],'NumeroDeVentas':['mean'],'Monetario':['mean'],'Devolucion':['mean'],'TicketPromedio':['mean'],'Monetario6M':['mean']})

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

clusterMotos = list(groupbyMotos.index)
countMotos = groupbyMotos['Recencia']['count']
recenciaMotos = groupbyMotos['Recencia']['mean']
cardifMotos = groupbyMotos['CARDIF']['mean']
numeroventasMotos = groupbyMotos['NumeroDeVentas']['mean']
monetarioMotos = groupbyMotos['Monetario']['mean']
devolucionMotos = groupbyMotos['Devolucion']['mean']
ticketMotos = groupbyMotos['TicketPromedio']['mean']
monetario6mMotos = groupbyMotos['Monetario6M']['mean']

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
    StructField("Segmentos", StringType(), True),
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
    StructField("Segmentos", StringType(), True),
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
    StructField("Segmentos", StringType(), True),
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

print('Hello World!')

# COMMAND ----------


