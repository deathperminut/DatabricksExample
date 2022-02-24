# Databricks notebook source
import os
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_theme(style="darkgrid")
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
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
raw_data = df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC # EDA

# COMMAND ----------

dataset = raw_data.copy()

# COMMAND ----------

dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ### Raw Data Distribution

# COMMAND ----------

dataset.describe()

# COMMAND ----------

def print_hist(data,special=""):
    fig, axs = plt.subplots(1,3,figsize=(15,5))
    col = ['Recency','Frequency','Monetary']
    color = ['tab:blue','red','orange']
    for i in range(3):
        name = special+str(col[i])
        axs[i].hist(data[name],bins=50,color=color[i])
        axs[i].set_title(name)
    fig.tight_layout()

# COMMAND ----------

print_hist(dataset,"")

# COMMAND ----------

def print_boxplot(data,special=""):
    fig, axs = plt.subplots(3,1,figsize=(12,5))
    col = ['Recency','Frequency','Monetary']
    color = ['tab:blue','red','orange']
    for i in range(3):
        sns.boxplot(x=col[i],data=data,ax=axs[i],color=color[i])
    fig.tight_layout()

# COMMAND ----------

print_boxplot(dataset,"")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Drop Outliers

# COMMAND ----------

def detect_outlier(data):
    outliers = []
    threshold= 3
    mean = np.mean(data)
    std =np.std(data)
    return min([y for y in data if np.abs( (y-mean) / std ) > threshold], default=1e18)

# COMMAND ----------

outlier_limits = {c:detect_outlier(dataset[c]) for c in dataset.columns}
print('Outliers Lower Limit')
outlier_limits

# COMMAND ----------

dataset_outlier = dataset[(dataset['Recency']>=outlier_limits['Recency'])
                        |(dataset['Frequency']>=outlier_limits['Frequency'])
                        |(dataset['Monetary']>=outlier_limits['Monetary'])]

# COMMAND ----------

#Drop Outliers
dataset.drop(dataset_outlier.index,inplace=True)
dataset.reset_index(inplace=True,drop=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data distribution after dropping outliers

# COMMAND ----------

print_hist(dataset,"")

# COMMAND ----------

print_boxplot(dataset,"")

# COMMAND ----------

#dataset['Monetary'] = (dataset['Monetary'] / (5*1e5)).apply(np.ceil) * 5*1e5

# COMMAND ----------

# MAGIC %md
# MAGIC ### Correlation between variables

# COMMAND ----------

correlation_matrix = dataset.corr()
plt.figure(figsize=(8,8),facecolor='white')
sns.heatmap(correlation_matrix,square=True,annot=True,cmap="RdYlGn")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Normalizing data

# COMMAND ----------

def normalize_data(data, columns):
    scaler = preprocessing.MinMaxScaler()
    d = scaler.fit_transform(data[columns])
    scaled_df = pd.DataFrame(d, columns=columns)
    for c in columns :
        data['Norm'+c] = scaled_df[c]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Elbow Method

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

# MAGIC %md
# MAGIC # KMeans

# COMMAND ----------

def kmeans_(data,n,col):
    #initiliaze model
    kmeans = KMeans(n_clusters=n,random_state=123,max_iter=800,n_init=30,algorithm='full')
    #fitting
    data['Cluster'] = kmeans.fit_predict(data[col])

# COMMAND ----------

def plot_2d(data,col):
    fig, axs = plt.subplots(3,3,figsize=(12,12))
    for i in range(3):
        for j in range(3):
            axs[i, j].scatter(data[col[i]],data[col[j]],c=data['Cluster'],cmap='viridis',s=5)
            axs[i, j].set_xlabel(col[i])
            axs[i, j].set_ylabel(col[j])
    fig.tight_layout()  

# COMMAND ----------

def plot_3d(data,col):
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(111, projection='3d') 
    # plot points
    ax.scatter(data[col[0]],
                data[col[1]],
                data[col[2]], 
                c=data["Cluster"],
                cmap='viridis',
                s=5,
                alpha=1)
    ax.view_init()
    plt.xlabel(col[0])
    plt.ylabel(col[1])
    ax.set_zlabel(col[2])
    plt.show()

# COMMAND ----------

data_rfm = dataset.copy()

# COMMAND ----------

normalize_data(data_rfm,['Recency','Frequency','Monetary'])

# COMMAND ----------

elbow_method(data_rfm,['NormRecency','NormFrequency','NormMonetary'])

# COMMAND ----------

kmeans_(data_rfm,4,['NormRecency','NormFrequency','NormMonetary'])

# COMMAND ----------

rst = data_rfm.groupby('Cluster').agg({'Recency':['min','max','mean'],
                                'Frequency':['min','max','mean'],
                                'Monetary':['min','max','mean'],
                                'NormRecency':['min','max'],
                                'NormFrequency':['min','max'],
                                'NormMonetary':['min','max','size']})
rst

# COMMAND ----------

data_sample = data_rfm.sample(int(data_rfm.shape[0]*0.1))

# COMMAND ----------

plot_2d(data_sample,['Recency','Frequency','Monetary'])

# COMMAND ----------

plot_3d(data_sample,['Recency','Frequency','Monetary'])

# COMMAND ----------

# MAGIC %md
# MAGIC # KMeans with Score

# COMMAND ----------

data_rfm = dataset.copy()

# COMMAND ----------

# data_drop_old = data_rfm[data_rfm['Recency'] >= 49]
# data_rfm.drop(data_drop_old.index,inplace=True)
# data_rfm.reset_index(inplace=True,drop=True)

# COMMAND ----------

data_rfm['ScoreRecency'] = pd.cut(
                            data_rfm['Recency'], 
                            bins=[-1, 24, data_rfm['Recency'].max()],
                            labels=[4, 1]).astype('int')
    
data_rfm['ScoreFrequency'] = pd.cut(
                                data_rfm['Frequency'], 
                                bins=[np.percentile(data_rfm['Frequency'],i) if (i != 0) else -1 for i in range(0,125,25)], 
                                labels=[1, 2, 3, 4]).astype('int')
    
data_rfm['ScoreMonetary'] = pd.cut(
                                data_rfm['Monetary'], 
                                bins=[np.percentile(data_rfm['Monetary'],i) if (i != 0) else -1 for i in range(0,125,25)], 
                                labels=[1, 2, 3, 4]).astype('int')

# COMMAND ----------

#Recency 
print([-1, 24, data_rfm['Recency'].max()])
    
#Frequency
print([np.percentile(data_rfm['Frequency'],i) if (i != 0) else -1 for i in range(0,125,25)])
    
#Monetary
print([np.percentile(data_rfm['Monetary'],i) if (i != 0) else -1 for i in range(0,125,25)])

# COMMAND ----------

data_rfm.describe()

# COMMAND ----------

elbow_method(data_rfm,['ScoreRecency','ScoreFrequency','ScoreMonetary'])

# COMMAND ----------

kmeans_(data_rfm,5,['ScoreRecency','ScoreFrequency','ScoreMonetary'])

# COMMAND ----------

rst = data_rfm.groupby('Cluster').agg({'Recency':['min','max','mean'],
                                'Frequency':['min','max','mean'],
                                'Monetary':['min','max','mean'],
                                'ScoreRecency':['min','max'],
                                'ScoreFrequency':['min','max'],
                                'ScoreMonetary':['min','max','size']})
rst

# COMMAND ----------

data_sample = data_rfm.sample(int(data_rfm.shape[0]*0.1))

# COMMAND ----------

plot_2d(data_sample,['Recency','Frequency','Monetary'])

# COMMAND ----------

plot_3d(data_sample,['Recency','Frequency','Monetary'])
