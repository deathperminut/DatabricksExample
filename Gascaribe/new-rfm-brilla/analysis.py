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


