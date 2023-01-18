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

rawData

# COMMAND ----------

sns.displot(rawData,x='Recency',aspect=20/8.27,kind='hist',bins=30,color='#EAAB00',fill=True)
plt.axvline(x=48, color='r', linestyle='-',label='Linea de 48 meses')
plt.ticklabel_format(style='plain')
plt.xticks(np.arange(0,max(rawData['Recency'])+1,10),rotation=45)
plt.title('Distribucion de Recencias en Clientes Brilla')
plt.legend()
plt.show() 

# COMMAND ----------

sns.displot(rawData,x='Monetary',aspect=20/8.27,kind='kde',color='#fa5d20',fill=True)
plt.ticklabel_format(style='plain')
plt.xticks(np.arange(0,max(rawData['Monetary'])+1,5000000),rotation=45)
plt.title('Distribucion de Monetario en Clientes Brilla')
plt.show() 

# COMMAND ----------

sns.displot(rawData,x='Frequency',aspect=20/8.27,kind='hist',bins=30,color='#EAAB00',fill=True)
plt.ticklabel_format(style='plain')
plt.xticks(np.arange(0,max(rawData['Frequency'])+1,10),rotation=45)
plt.title('Distribucion de Frecuencias en Clientes Brilla')
plt.show() 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### **Data Processing**

# COMMAND ----------

def modelo1(df):

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
    
  return df

# COMMAND ----------

def get_inactivos(df):
    inactivos = df[df['Recency'] > 49]
    
    inactivos_df = pd.DataFrame({'Identificacion':inactivos['Identificacion'],
                                 'Recency':inactivos['Recency'],
                                 'Frequency':inactivos['Frequency'],
                                 'Monetary':inactivos['Monetary'],
                                'cluster':len(inactivos)*[-1],
                                'name':len(inactivos)*['Inactivo']})
    print(f'Numero de Usuarios Inactivos: {len(inactivos_df)}')
    return inactivos_df

# COMMAND ----------

inactivos_df = get_inactivos(rawData)
inactivos_df.head()

# COMMAND ----------

from mpl_toolkits.mplot3d import Axes3D


def get_sample(df):

    df['RFM'] = df[['Recency-Score','Frequency-Score','Monetary-Score']].astype(str).agg(''.join,axis=1)

    X_new = []
    for i in df['RFM'].unique():
        X_new.append(df[df['RFM'] == i].head(1).index.values)

    new_X = [int(str(x)[1:-1]) for x in X_new]

    X = df.iloc[new_X]

    return X

def plot3d(X):
  colors = ['#DF2020', '#81DF20', '#2095DF','#F4D03F']
  kmeans = KMeans(n_clusters=4, random_state=0)
  X['cluster'] = kmeans.fit_predict(X[['Recency-Score','Monetary-Score','Frequency-Score']])
  X['c'] = X.cluster.map({0:colors[0], 1:colors[1], 2:colors[2],3:colors[3]})
  fig = plt.figure(figsize=(40,10))
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(X['Recency-Score'], X['Monetary-Score'], X['Frequency-Score'], c=X.c, s=15)
  ax.set_xlabel('Recency')
  ax.set_ylabel('Monetary')
  ax.set_zlabel('Frequency')
  
  #for angle in range(0,360):
   # ax.view_init(30,angle)
   # plt.draw()
   # plt.pause(0.001)
  plt.show()
  return kmeans

# COMMAND ----------

rawData['Identificacion'] = rawData['Identificacion']
print(df.head())
#df = df[df['Recency'] <= 49]

X = modelo1(rawData)

#X = get_sample(X)

kmeans = plot3d(X)

# COMMAND ----------

centroids = kmeans.cluster_centers_
clusters = pd.DataFrame(centroids, columns=['Recency-Score','Monetary-Score','Frequency-Score'])

clusters['cluster'] = kmeans.predict(clusters[['Recency-Score','Monetary-Score','Frequency-Score']]) 
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

XMerged

# COMMAND ----------

mergedX = pd.concat([XMerged[['Identificacion','Recency','Frequency','Monetary','cluster','name']],inactivos_df[['Identificacion','Recency','Frequency','Monetary','cluster','name']]],axis=0)

# COMMAND ----------

mergedX.groupby('name').agg({'Recency':['count','mean'],'Frequency':['mean'],'Monetary':['mean']})

# COMMAND ----------


