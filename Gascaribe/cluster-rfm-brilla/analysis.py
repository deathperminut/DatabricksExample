# Databricks notebook source
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer
import time 
from scipy import stats
from scipy.stats import skew 
from scipy.stats import norm

from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import metrics
from scipy.spatial.distance import cdist
from mpl_toolkits import mplot3d

import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# COMMAND ----------

# Constants
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

# Data Ingestion
query = 'SELECT * FROM ModeloRFMBrilla.BaseRFM'

df = spark.read \
  .format("com.databricks.spark.sqldw") \
  .option("url", sqlDwUrl) \
  .option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
  .option("forwardSparkAzureStorageCredentials", "true") \
  .option("maxStrLength", "1024" ) \
  .option("query", query) \
  .load()

df = df.drop('Identificacion')
dataset = df.toPandas()

# COMMAND ----------

dataset.shape
dataset.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA

# COMMAND ----------

# MAGIC %md
# MAGIC ### Features Distribution

# COMMAND ----------

sns.set_theme(style="darkgrid")
fig, axs = plt.subplots(1,3,figsize=(15,5))
axs[0].hist(dataset['Recency'],bins=50)
axs[0].set_title('Recency')
axs[1].hist(dataset['Frequency'],bins=50,color='red',alpha=0.6)
axs[1].set_title('Frequency')
axs[2].hist(dataset['Monetary'],bins=50,color='orange',alpha=0.6)
axs[2].set_title('Monetary')
fig.tight_layout()

# COMMAND ----------

fig, axs = plt.subplots(3,1,figsize=(16,8))
sns.boxplot(x='Recency',data=dataset,ax=axs[0])
sns.boxplot(x='Frequency',data=dataset,ax=axs[1])
sns.boxplot(x='Monetary',data=dataset,ax=axs[2])
fig.tight_layout()

# COMMAND ----------

#Drop Outliers
dataset.drop(dataset[dataset['Frequency'] > 10].index,inplace=True)
dataset.drop(dataset[dataset['Monetary'] > 1.5e7].index,inplace=True)

# COMMAND ----------

dataset.reset_index(inplace=True,drop=True)

# COMMAND ----------

dataset_org_feat = dataset.copy()

# COMMAND ----------

sns.set_theme(style="darkgrid")
fig, axs = plt.subplots(3,1,figsize=(16,8))
sns.boxplot(x='Recency',data=dataset,ax=axs[0])
sns.boxplot(x='Frequency',data=dataset,ax=axs[1])
sns.boxplot(x='Monetary',data=dataset,ax=axs[2])
fig.tight_layout()

# COMMAND ----------

fig, axs = plt.subplots(1,3,figsize=(15,5))
axs[0].hist(dataset['Recency'],bins=50)
axs[0].set_title('Recency')
axs[1].hist(dataset['Frequency'],bins=50,color='red',alpha=0.6)
axs[1].set_title('Frequency')
axs[2].hist(dataset['Monetary'],bins=50,color='orange',alpha=0.6)
axs[2].set_title('Monetary')
fig.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Features Correlation

# COMMAND ----------

fig, axs = plt.subplots(3,3,figsize=(12,12))
c = dataset.columns
for i in range(3):
    for j in range(3):
        axs[i, j].plot(dataset[c[i]],dataset[c[j]],'x',alpha=0.6)
        axs[i, j].set_xlabel(c[i])
        axs[i, j].set_ylabel(c[j])
fig.tight_layout()    

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Transformation

# COMMAND ----------

#Normalizing Features
numerical_features = dataset.dtypes[dataset.dtypes != 'object'].index
numerical = dataset[numerical_features]
scaler = preprocessing.MinMaxScaler()
d = scaler.fit_transform(numerical)
scaled_df = pd.DataFrame(d, columns=numerical_features)
for c in numerical_features :
    dataset[c] = scaled_df[c]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Elbow Method

# COMMAND ----------

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)
 
for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(dataset)
    kmeanModel.fit(dataset)
 
    distortions.append(sum(np.min(cdist(dataset, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / dataset.shape[0])
    inertias.append(kmeanModel.inertia_)
 
    mapping1[k] = sum(np.min(cdist(dataset, kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / dataset.shape[0]
    mapping2[k] = kmeanModel.inertia_

# COMMAND ----------

for key, val in mapping1.items():
    print(f'{key} : {val}')

# COMMAND ----------

model = KMeans()
visualizer = KElbowVisualizer(model, k=(0,12))
visualizer.fit(dataset)
visualizer.show() 

# COMMAND ----------

plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()

# COMMAND ----------

for key, val in mapping2.items():
    print(f'{key} : {val}')

# COMMAND ----------

plt.plot(K, inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modeling

# COMMAND ----------

# MAGIC %md
# MAGIC ### KMeans - 5 Clusters

# COMMAND ----------

model = KMeans(n_clusters=5)

# COMMAND ----------

fig, axs = plt.subplots(3,3,figsize=(12,12))
c = dataset.columns
for i in range(3):
    for j in range(3):
        axs[i, j].scatter(dataset_org_feat[c[i]],dataset_org_feat[c[j]],c=dataset_org_feat['label'],cmap='viridis')
        axs[i, j].set_xlabel(c[i])
        axs[i, j].set_ylabel(c[j])
fig.tight_layout()  

# COMMAND ----------

dataset_org_feat.groupby('label').agg({"Recency":["min","max"], "Frequency":["min","max"], "Monetary": ["min","max"]})

# COMMAND ----------

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d') 
ax.scatter(dataset_org_feat["Recency"],
                dataset_org_feat["Monetary"],
                dataset_org_feat["Frequency"], 
                c=dataset_org_feat["label"],
                cmap='viridis',
                s=20,
                alpha=0.6)
plt.xlabel("Recency")
plt.ylabel("Monetary")
ax.set_zlabel("Frequency")
plt.show()

# COMMAND ----------

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d') 
ax.scatter(dataset_org_feat["Recency"],
                dataset_org_feat["Monetary"],
                dataset_org_feat["Frequency"], 
                c=dataset_org_feat["label"],
                cmap='viridis',
                s=20,
                alpha=0.6)
ax.view_init(30, 185)
plt.xlabel("Recency")
plt.ylabel("Monetary")
ax.set_zlabel("Frequency")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### KMeans - 4 Clusters

# COMMAND ----------

dataset.drop(dataset_org_feat[dataset_org_feat['Recency'] > 730].index,inplace=True)
dataset_org_feat.drop(dataset_org_feat[dataset_org_feat['Recency'] > 730].index,inplace=True)

# COMMAND ----------

model = KMeans(n_clusters=4)
dataset['label'] = model.fit_predict(dataset)
dataset_org_feat['label'] = dataset['label']

# COMMAND ----------

fig, axs = plt.subplots(3,3,figsize=(12,12))
c = dataset.columns
for i in range(3):
    for j in range(3):
        axs[i, j].scatter(dataset_org_feat[c[i]],dataset_org_feat[c[j]],c=dataset_org_feat['label'],cmap='viridis')
        axs[i, j].set_xlabel(c[i])
        axs[i, j].set_ylabel(c[j])
fig.tight_layout()  

# COMMAND ----------

dataset_org_feat.groupby('label').agg({"Recency":["min","max"], "Frequency":["min","max"], "Monetary": ["min","max"]})

# COMMAND ----------

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d') 
ax.scatter(dataset_org_feat["Recency"],
                dataset_org_feat["Monetary"],
                dataset_org_feat["Frequency"], 
                c=dataset_org_feat["label"],
                cmap='viridis',
                s=20,
                alpha=0.6)
plt.xlabel("Recency")
plt.ylabel("Monetary")
ax.set_zlabel("Frequency")
plt.show()

# COMMAND ----------

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d') 
ax.scatter(dataset_org_feat["Recency"],
                dataset_org_feat["Monetary"],
                dataset_org_feat["Frequency"], 
                c=dataset_org_feat["label"],
                cmap='viridis',
                s=20,
                alpha=0.6)
ax.view_init(30, 185)
plt.xlabel("Recency")
plt.ylabel("Monetary")
ax.set_zlabel("Frequency")
plt.show()
