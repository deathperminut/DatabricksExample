# Databricks notebook source
import os
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType

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
query = 'SELECT * FROM MPAlivios.Datos WHERE Entrenamiento = ' + ('1' if is_training else '0')

df = spark.read \
  .format("com.databricks.spark.sqldw") \
  .option("url", sqlDwUrl) \
  .option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
  .option("forwardSparkAzureStorageCredentials", "true") \
  .option("maxStrLength", "1024" ) \
  .option("query", query) \
  .load()

df = df.drop('FechaCierre')
df = df.drop('Entrenamiento')

data = df.toPandas()

# COMMAND ----------

# Obtain feature values
if is_training:
  loc_df = spark.read \
    .format("com.databricks.spark.sqldw") \
    .option("url", sqlDwUrl) \
    .option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
    .option("forwardSparkAzureStorageCredentials", "true") \
    .option("maxStrLength", "1024" ) \
    .option("query", "SELECT DISTINCT Localidad FROM Cartera.FactResumenCierreDia") \
    .load()
  loc_data = loc_df.toPandas()
  localities = loc_data['Localidad'].unique()

# Data Preparation
features = data
  
# Prefiltering
features = features[features['Categoria'] == 1]

# Select columns
features = features.drop([
  'Año', 'Mes', 'Dia', 'Categoria', \
  'Barrio', 'Departamento', 'FinanciacionMayo', \
  'FinanciacionJunio', 'FinanciacionJulio', 'RecaudoMayo', \
  'RecaudoJunio', 'RecaudoJulio', 'RangoEdadMoraFinal', 'MoraMaxima'], axis = 1)

if is_training:
  features = features.drop(['Contrato'], axis = 1)
  
# Class condensation
features['RangoEdadMora'] = features['RangoEdadMora'].apply(lambda v: 480 if v > 480 else v)

# Outlier Removal
if is_training:
  features = features[features['DeudaCorrienteNoVencidaGas'] <= 2.5 * 1e6]
  features = features[features['DeudaCorrienteVencidaGas'] <= 5 * 1e6]
  features = features[features['DeudaDiferidaGas'] <= 20 * 1e6]
  features = features[features['DeudaCorrienteNoVencidaBrilla'] <= 1.2 * 1e6]
  features = features[features['DeudaCorrienteVencidaBrilla'] <= 3 * 1e6]
  features = features[features['DeudaDiferidaBrilla'] <= 10 * 1e6]
  features = features[features['DeudaCorrienteNoVencidaOtros'] <= 0.1 * 1e6]
  features = features[features['DeudaCorrienteVencidaOtros'] <= 0.5 * 1e6]
  features = features[features['Cuota'] <= 1.5 * 1e6]
  sample_size = int(0.8e6)
  if len(features) > sample_size:
    features = features.sample(n = sample_size)
  
# Output variable
if is_training:
  features['Pago'] = features.apply(lambda x: (1 if x['EdadMoraFinal'] == 0 else 0) if x['EdadMora'] == 0 else (1 if x['EdadMoraFinal'] - x['EdadMora'] < 28 else 0), axis = 1)

features = features.drop(['EdadMora', 'EdadMoraFinal'], axis = 1)

# COMMAND ----------

one_hot_columns = ['Localidad', 'Subcategoria', 'RangoEdadMora', 'Refinanciado']
if is_training:
  # Separating
  X_train = features.drop(['Pago'], axis = 1)
  y_train = features['Pago']
  # One Hot Encoding
  categories = [localities, [i for i in range(1, 6 + 1)], [i for i in range(0, 480 + 30, 30)], ['SI', 'NO']]
  encoder = OneHotEncoder(drop = 'first', sparse = False, categories = categories)
  X_train = np.concatenate(( \
    X_train.drop(one_hot_columns, axis = 1).to_numpy(), \
    encoder.fit_transform(X_train[one_hot_columns]) \
  ), axis = 1)
  # Scaling
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  # Balancing
  sm = RandomOverSampler(ratio = 1.0)
  X_train, y_train = sm.fit_sample(X_train, y_train)
  # Model training
  model = RandomForestClassifier() # GradientBoostingClassifier(learning_rate = 0.1, n_estimators = 100, min_impurity_decrease = 0.2, max_depth = 3)
  model.fit(X_train, y_train)
  # Dump model parts
  with open('/dbfs/FileStore/tables/MPAlivios_Encoder.pkl', 'wb') as handle:
    pkl.dump(encoder, handle, protocol = pkl.HIGHEST_PROTOCOL)
  with open('/dbfs/FileStore/tables/MPAlivios_Scaler.pkl', 'wb') as handle:
    pkl.dump(scaler, handle, protocol = pkl.HIGHEST_PROTOCOL)
  with open('/dbfs/FileStore/tables/MPAlivios_Model.pkl', 'wb') as handle:
    pkl.dump(model, handle, protocol = pkl.HIGHEST_PROTOCOL)
else:
  # Load model partsparts
  with open('/dbfs/FileStore/tables/MPAlivios_Encoder.pkl', 'rb') as handle:
    encoder = pkl.load(handle)
  with open('/dbfs/FileStore/tables/MPAlivios_Scaler.pkl', 'rb') as handle:
    scaler = pkl.load(handle)
  with open('/dbfs/FileStore/tables/MPAlivios_Model.pkl', 'rb') as handle:
    model = pkl.load(handle)
  # Separating
  id_list = features['Contrato']
  X_pred = features.drop(['Contrato'], axis = 1)
  # One Hot Encoding
  encoded_cols = encoder.transform(X_pred[one_hot_columns])
  non_encoded_cols = X_pred.drop(one_hot_columns, axis = 1).to_numpy()
  X_pred = np.concatenate((non_encoded_cols, encoded_cols), axis = 1)
  # Scaling
  X_pred = scaler.transform(X_pred)
  # Model Prediction
  y_pred = model.predict_proba(X_pred)[:, 1]
  # Save Predictions
  dataPandas = pd.DataFrame({ 'Contrato': id_list, 'ProbabilidadPago': y_pred })
  schema = StructType([
    StructField("Contrato", IntegerType(), True),
    StructField("ProbabilidadPago", FloatType(), True),
  ])
  df = spark.createDataFrame(dataPandas, schema = schema)
  df.write \
    .format("com.databricks.spark.sqldw") \
    .option("url", sqlDwUrl) \
    .option("forwardSparkAzureStorageCredentials", "true") \
    .option("dbTable", "MPAlivios.Resultados") \
    .option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
    .mode("overwrite") \
    .save()
