# Databricks notebook source
from databricks import feature_store
import pandas as pd
from pyspark.sql.functions import *

# COMMAND ----------

dwDatabase = dbutils.secrets.get(scope='efigas', key='dwh-name')
dwServer = dbutils.secrets.get(scope='efigas', key='dwh-host')
dwUser = dbutils.secrets.get(scope='efigas', key='dwh-user')
dwPass = dbutils.secrets.get(scope='efigas', key='dwh-pass')
dwJdbcPort = dbutils.secrets.get(scope='efigas', key='dwh-port')
dwJdbcExtraOptions = ""
sqlDwUrl = "jdbc:sqlserver://" + dwServer + ".database.windows.net:" + dwJdbcPort + ";database=" + dwDatabase + ";user=" + dwUser + ";password=" + dwPass + ";" + dwJdbcExtraOptions
storage_account_name = dbutils.secrets.get(scope='efigas', key='bs-name')
blob_container = dbutils.secrets.get(scope='efigas', key='bs-container')
blob_storage = storage_account_name + ".blob.core.windows.net"
config_key = "fs.azure.account.key."+storage_account_name+".blob.core.windows.net"
blob_access_key = dbutils.secrets.get(scope='efigas', key='bs-access-key')
spark.conf.set(config_key, blob_access_key)

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

# COMMAND ----------

rawData = df.toPandas()

# COMMAND ----------

def preprocess_inputs(df):

    df = df.copy()
    df = df[df['Recency'] <= 49]
    df = df[df['Frequency'] != 0]
    df['Identificacion'] = df['Identificacion'].replace('-','').astype('int')
   
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
X['Score'] = X['Recency-Score'].astype('str') + X['Frequency-Score'].astype('str') + X['Monetary-Score'].astype('str')
X.head()

# COMMAND ----------

def get_inactivos(df):
    inactivos = df[df['Recency'] > 49]
    
    inactivos_df = pd.DataFrame({'Identificacion':inactivos['Identificacion'],
                                 'Recency':inactivos['Recency'],
                                 'Frequency':inactivos['Frequency'],
                                 'Monetary':inactivos['Monetary'],
                                'cluster':len(inactivos)*[5],
                                'name':len(inactivos)*['Inactivos'],
                                'Score':len(inactivos)*['000']})
    print(f'Numero de Usuarios Inactivos: {len(inactivos_df)}')
    return inactivos_df

# COMMAND ----------

inactivos_df = get_inactivos(rawData)
inactivos_df.head()

# COMMAND ----------

fs = feature_store.FeatureStoreClient()
table_name = f"efg_segmentacion_features"
inactivos = "EFG_inactivos_brilla"

# COMMAND ----------

#Creacion/actualización de feature store de clientes activos de brilla
sparkDF=spark.createDataFrame(X)
try:
    fs.write_table(
        name=table_name,
        df=sparkDF,
        mode='overwrite'
    )
except:
    fs.create_table(
        name=table_name,
        primary_keys=["Identificacion"],
        schema=sparkDF.schema,
        description="EFG_Segmentacion_Features"
    )


# COMMAND ----------

#Creacion/actualización de feature store de clientes inactivos de brilla
spark_inactivos = spark.createDataFrame(inactivos_df)
try:
    fs.write_table(
        name=inactivos,
        df=spark_inactivos,
        mode='overwrite'
    )
except:
    fs.create_table(
        name=inactivos,
        primary_keys=["Identificacion"],
        schema=spark_inactivos.schema,
        description="EFG_inactivos_brilla"
    )
