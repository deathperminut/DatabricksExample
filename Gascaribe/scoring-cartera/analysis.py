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

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

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

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC ### **Queries**

# COMMAND ----------

query = "SELECT * FROM ScoringCartera.FactScoringResidencialNuevo"

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
    
newdf = pd.concat([df6,df12,df24],axis=0).reset_index(drop=True)

# COMMAND ----------

newdf

# COMMAND ----------

ponderado = []
for i in range(len(newdf)):
    if newdf['IdTipoProducto'][i] == 7014:
        xPago = 0.15
        xMora = 0.35
        xRefinanciaciones = 0.25
        xSuspensiones = 0.25

        ponderado.append(xPago*newdf['VarPago'][i] + xMora*newdf['VarMora'][i] + xRefinanciaciones*newdf['VarRefinanciaciones'][i] + xSuspensiones*newdf['varSuspensiones'][i])
    else:
        xPago = 0.14
        xMora = 0.53
        xRefinanciaciones = 0.33

        ponderado.append(xPago*newdf['VarPago'][i] + xMora*newdf['VarMora'][i] + xRefinanciaciones*newdf['VarRefinanciaciones'][i])
        
newdf['Ponderado'] = ponderado

# COMMAND ----------

newdf

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

df[(df['IdTipoProducto'] == 7014) & (df['Intervalo'] == 6)]['Segmento'].value_counts().values.sum()

# COMMAND ----------

df[(df['IdTipoProducto'] == 7055) & (df['Intervalo'] == 6)]['Segmento'].value_counts().values.sum()

# COMMAND ----------

df[(df['IdTipoProducto'] == 7014) & (df['Intervalo'] == 24)]['Segmento'].value_counts().sort_index(ascending=True)

# COMMAND ----------

df[(df['IdTipoProducto'] == 7055) & (df['Intervalo'] == 24)]['Segmento'].value_counts().sort_index(ascending=True)

# COMMAND ----------

df['SegmentoNombre'] = df['Segmento'].replace({
    0:'Pesimo',
    1:'Malo',
    2:'Regular',
    3:'Bueno',
    4:'Excelente'
})

# COMMAND ----------

df.head()

# COMMAND ----------

fig, ax = plt.subplots(1,1)

intervalo = 24
sns.barplot(x=df[(df['IdTipoProducto'] == 7014) & (df['Intervalo'] == intervalo)]['Segmento'].value_counts().sort_index(ascending=True).keys(),y=df[(df['IdTipoProducto'] == 7055) & (df['Intervalo'] == intervalo)]['Segmento'].value_counts().sort_index(ascending=True).values)

ax.set_xticklabels(['Pesimo','Malo','Regular','Bueno','Excelente'], rotation='vertical', fontsize=18)
plt.title(f'Numero de Productos Brilla por Segmento. Intervalo: {intervalo}')

# COMMAND ----------

df[(df['IdTipoProducto'] == 7055) & (df['Intervalo'] == 24)]['Segmento'].value_counts().sort_index(ascending=True)

# COMMAND ----------

df[df['IdProducto'] == 52078666]

# COMMAND ----------

df[(df['IdTipoProducto'] == 7014)]['Ponderado'].min()

# COMMAND ----------

df[df['Ponderado'] == 0.31624337875034825]

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### **Elbow Method**

# COMMAND ----------

def elbow_method(data, columns):
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    data_em = data[columns]
    K = range(1, 10)
 
    for k in K:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k).fit(data_em)
        kmeanModel.fit(data_em)
     
        distortions.append(sum(np.min(cdist(data_em, kmeanModel.cluster_centers_,
                                            'euclidean'), axis=1)) / data_em.shape[0])
        inertias.append(kmeanModel.inertia_)
 
        mapping1[k] = sum(np.min(cdist(data_em, kmeanModel.cluster_centers_,
                                       'euclidean'), axis=1)) / data_em.shape[0]
        mapping2[k] = kmeanModel.inertia_
        
    for key, val in mapping1.items():
        print(f'{key} : {val}')
        
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method using Distortion')
    plt.show()
    
    for key, val in mapping2.items():
        print(f'{key} : {val}')
    
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method using Inertia')
    plt.show()

# COMMAND ----------

elbow_method(df,['Ponderado'])

# COMMAND ----------

# BIC for GMM
from sklearn.mixture import GaussianMixture
n_components = range(1, 10)
covariance_type = ['spherical', 'tied', 'diag', 'full']
score=[]
for cov in covariance_type:
    for n_comp in n_components:
        gmm=GaussianMixture(n_components=n_comp,covariance_type=cov)
        gmm.fit(np.array(df['Ponderado']).reshape(-1,1))
        score.append((cov,n_comp,gmm.bic(np.array(df['Ponderado']).reshape(-1,1))))
score

# COMMAND ----------

score[18:27]

# COMMAND ----------

spherical = {}
tied = {}
diag = {}
full = {}
for i in range(len(score)):
    if i <= 8:
        spherical[score[i][1]] = score[i][2]
    elif i > 8 and i <= 17:
        tied[score[i][1]] = score[i][2]
    elif i > 17 and i <= 26:
        diag[score[i][1]] = score[i][2]
    else:
        full[score[i][1]] = score[i][2]

# COMMAND ----------

scores_dict = {
    #'Spherical':spherical,
    'Tied':tied
    #'Diagonal':diag,
    #'Full':full
}

for name,s in scores_dict.items():
    plt.plot(s.keys(),s.values(), linestyle='--', marker='o', color='b')
    plt.title(f'{name} score')
    plt.show()

# COMMAND ----------

pd.DataFrame(score,columns=['Method','Number of Clusters','Score'])

# COMMAND ----------

sns.histplot(df[(df['IdTipoProducto'] == 7055) & (df['Intervalo'] == 12)]['Ponderado'],bins=6)

# COMMAND ----------

sns.histplot(df[(df['IdTipoProducto'] == 7014) & (df['Intervalo'] == 12) & (df['Ponderado'] != 1)]['Ponderado'],bins=4)

# COMMAND ----------

len(df[(df['IdTipoProducto'] == 7014) & (df['Ponderado'] == 1) & (df['Intervalo'] == 6)])

# COMMAND ----------

len(df[(df['IdTipoProducto'] == 7014) & (df['Ponderado'] != 1) & (df['Intervalo'] == 6)])

# COMMAND ----------

len(df[(df['IdTipoProducto'] == 7055) & (df['Ponderado'] == 1)])

# COMMAND ----------

np.histogram(df[(df['Ponderado'] != 1)]['Ponderado'],bins=4)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### **Writing Into DWH**

# COMMAND ----------

df.head()

# COMMAND ----------

results = df[['IdTipoProducto','IdProducto','Intervalo','VarPago','VarMora','VarRefinanciaciones','varSuspensiones','Castigado','Ponderado','Segmento','SegmentoNombre']]
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
    StructField("Castigado", IntegerType(), True),
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


