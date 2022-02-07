# Databricks notebook source
import os
import pandas as pd
import numpy as np
import pickle as pkl
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType

from sklearnex import patch_sklearn
patch_sklearn()

import sklearn
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

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
is_training = dbutils.widgets.get("is_training") == "true"

# COMMAND ----------

# Data Ingestion
query = 'SELECT * FROM ModeloPrediccionPago.DatosModelo WHERE Pago_Actual IN ' + ('(0, 1)' if is_training else '(-1)')

df = spark.read \
  .format("com.databricks.spark.sqldw") \
  .option("url", sqlDwUrl) \
  .option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
  .option("forwardSparkAzureStorageCredentials", "true") \
  .option("maxStrLength", "1024" ) \
  .option("query", query) \
  .load()

raw_data = df.toPandas()

# COMMAND ----------

dataset = raw_data.copy()

# COMMAND ----------

dataset.drop(['FechaEntrenamiento','FechaPrediccion','Producto','Cartera','DeudaCorrienteVencida'],axis=1,inplace=True)

# COMMAND ----------

# Cast columns
columns_to_float = ['EdadMora_30','EdadMora_60','EdadMora_90','Pago','Pago_Temprano']
for c in columns_to_float :
    dataset[c] = dataset[c].astype(float)
columns_to_str = ['Categoria','Refinanciado','EstadoFinanciero','EstadoCorte','EstadoProducto','TipoProducto']
for c in columns_to_str :
    dataset[c] = dataset[c].astype(str)

# COMMAND ----------

# Separate features from target variable
X = dataset.drop(['Pago_Actual'], axis = 1)
Y = dataset['Pago_Actual']
# Scale numerical features
numerical_features = X.dtypes[X.dtypes != 'object'].index
numerical = X[numerical_features]
scaler = MinMaxScaler()
d = scaler.fit_transform(numerical)
scaled_df = pd.DataFrame(d, columns=numerical_features)
for c in numerical_features :
  dataset[c] = scaled_df[c]

# COMMAND ----------

# Oversampling 
if is_training :
  ros = RandomOverSampler(random_state=123, sampling_strategy=0.8)
  X, Y = ros.fit_resample(X, Y)

# COMMAND ----------

one_hot_columns = ['Categoria','TipoProducto','Refinanciado','EstadoFinanciero','EstadoCorte','EstadoProducto']
if is_training :
  #One-hot enconding
  categories = [[101,102,103,104,105,106,201,202],
                [-1,3,6121,7014,7052,7053,7054,7055],
                [1,0],
                [1,2,3,4],
                [-1,1,2,3,5,6,92,94,95,96,99,100,110,111,112,113],
                ['Activo','Suspendido','Retirado sin instalación','Retirado']]
  encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, categories=categories)
  X_encoded = pd.DataFrame(encoder.fit_transform(X[one_hot_columns]))
  X_encoded.columns = encoder.get_feature_names(one_hot_columns)
  X.drop(one_hot_columns, axis=1, inplace=True)
  X = pd.concat([X, X_encoded], axis=1)
  # Model parameters
  params = {'bootstrap': True,
        'max_depth': 110,
        'min_samples_leaf': 3,
        'min_samples_split': 8,
        'n_estimators': 300}
  # Model fitting
  model = RandomForestClassifier(**params)
  model.fit(X, Y)
  # Dump
  with open('/dbfs/FileStore/tables/ModeloPrediccionPago_encoder.pkl', 'wb') as handle:
    pkl.dump(encoder, handle, protocol = pkl.HIGHEST_PROTOCOL)
  with open('/dbfs/FileStore/tables/ModeloPrediccionPago_model.pkl', 'wb') as handle:
    pkl.dump(model, handle, protocol = pkl.HIGHEST_PROTOCOL)

else:
  # Load
  with open('/dbfs/FileStore/tables/ModeloPrediccionPago_encoder.pkl', 'rb') as handle:
    encoder = pkl.load(handle)
  with open('/dbfs/FileStore/tables/ModeloPrediccionPago_model.pkl', 'rb') as handle:
    model = pkl.load(handle)
  # One-hot encoding
  X_encoded = pd.DataFrame(encoder.transform(X[one_hot_columns]))
  X_encoded.columns = encoder.get_feature_names(one_hot_columns)
  X.drop(one_hot_columns, axis=1, inplace=True)
  X = pd.concat([X, X_encoded], axis=1)
  # Save row id
  id_list = raw_data['Producto']
  # Make predictions
  Y_pred_proba = model.predict_proba(X)[:,1]
  # Save Predictions
  dataPandas = pd.DataFrame({ 'IdProducto': id_list, 'Probabilidad': Y_pred_proba})
  schema = StructType([
    StructField("IdProducto", IntegerType(), True),
    StructField("Probabilidad", FloatType(), True)
  ])
  df = spark.createDataFrame(dataPandas, schema = schema)
  df.write \
    .format("com.databricks.spark.sqldw") \
    .option("url", sqlDwUrl) \
    .option("forwardSparkAzureStorageCredentials", "true") \
    .option("dbTable", "Stage.ModeloPrediccionPagoFactPrediccion") \
    .option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
    .mode("overwrite") \
    .save()
