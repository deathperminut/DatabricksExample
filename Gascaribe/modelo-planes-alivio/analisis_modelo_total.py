# Databricks notebook source
import os
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
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
is_training = True # dbutils.widgets.get("is_training") == "true"

# COMMAND ----------

# Data Ingestion
query = 'SELECT * FROM MPAlivios.Datos2' # 'SELECT * FROM MPAlivios.Datos WHERE Entrenamiento = ' + ('1' if is_training else '0')

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

# Class condensation
features['RangoEdadMora'] = features['RangoEdadMora'].apply(lambda v: 480 if v > 480 else v)

# if is_training:
#   features = features.drop(['Contrato'], axis = 1)

# Outlier Removal
# if is_training:
  
# Output variable
# if is_training:
features['Pago'] = features.apply(lambda x: (1 if x['EdadMoraFinal'] == 0 else 0) if x['EdadMora'] == 0 else (1 if x['EdadMoraFinal'] - x['EdadMora'] < 28 else 0), axis = 1)
features = features.drop(['EdadMora', 'EdadMoraFinal'], axis = 1)

# COMMAND ----------

"""
one_hot_columns = ['Localidad', 'Subcategoria', 'RangoEdadMora', 'Refinanciado']
if is_training:
  # Separating
  X_train = features.drop(['Pago'], axis = 1)
  y_train = features['Pago']
  # One Hot Encoding
  encoder = OneHotEncoder(drop = 'first', sparse = False)
  encoded_cols = encoder.fit_transform(X_train[one_hot_columns])
  non_encoded_cols = X_train.drop(one_hot_columns, axis = 1).to_numpy()
  X_encoded = np.concatenate((non_encoded_cols, encoded_cols), axis = 1)
  # Scaling
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X_encoded)
  # Balancing
  sm = RandomOverSampler(random_state = 382, ratio = 1.0)
  X_sampled, y_sampled = sm.fit_sample(X_scaled, y_train)
  # Model training
  model = GradientBoostingClassifier(learning_rate = 0.1, n_estimators = 100, min_impurity_decrease = 0.2, max_depth = 3)
  model.fit(X_sampled, y_sampled)
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
  X_encoded = np.concatenate((non_encoded_cols, encoded_cols), axis = 1)
  # Scaling
  X_scaled = scaler.transform(X_encoded)
  # Model Prediction
  y_pred = model.predict_proba(X_scaled)[:, 1]
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
"""

# COMMAND ----------

one_hot_columns = ['Localidad', 'Subcategoria', 'RangoEdadMora', 'Refinanciado']

# COMMAND ----------

from sklearn.model_selection import train_test_split

# COMMAND ----------

train_features, test_features = train_test_split(features, train_size = 0.8)

# COMMAND ----------

train_features = train_features.sample(n = int(0.8e6))
train_features = train_features[train_features['DeudaCorrienteNoVencidaGas'] <= 2.5 * 1e6]
train_features = train_features[train_features['DeudaCorrienteVencidaGas'] <= 5 * 1e6]
train_features = train_features[train_features['DeudaDiferidaGas'] <= 20 * 1e6]
train_features = train_features[train_features['DeudaCorrienteNoVencidaBrilla'] <= 1.2 * 1e6]
train_features = train_features[train_features['DeudaCorrienteVencidaBrilla'] <= 3 * 1e6]
train_features = train_features[train_features['DeudaDiferidaBrilla'] <= 10 * 1e6]
train_features = train_features[train_features['DeudaCorrienteNoVencidaOtros'] <= 0.1 * 1e6]
train_features = train_features[train_features['DeudaCorrienteVencidaOtros'] <= 0.5 * 1e6]
train_features = train_features[train_features['Cuota'] <= 1.5 * 1e6]

# COMMAND ----------

X_train = train_features.drop(['Contrato', 'Pago'], axis = 1)
y_train = train_features['Pago']

# COMMAND ----------

encoder = OneHotEncoder(drop = 'first', sparse = False)
X_encoded = encoder.fit_transform(X_train[one_hot_columns])
X_non_encoded = X_train.drop(one_hot_columns, axis = 1)
X_train = np.concatenate((X_non_encoded, X_encoded), axis = 1)

# COMMAND ----------

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# COMMAND ----------

sm = RandomOverSampler(ratio = 1.0)
X_train, y_train = sm.fit_sample(X_train, y_train)

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# COMMAND ----------

# model = LogisticRegression(max_iter = 1000)
model = RandomForestClassifier()

# COMMAND ----------

model.fit(X_train, y_train)

# COMMAND ----------

from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

# COMMAND ----------

X_test = test_features.drop(['Contrato', 'Pago'], axis = 1)
y_test = test_features['Pago']
X_encoded = encoder.transform(X_test[one_hot_columns])
X_non_encoded = X_test.drop(one_hot_columns, axis = 1)
X_test = np.concatenate((X_non_encoded, X_encoded), axis = 1)
X_test = scaler.transform(X_test)

# COMMAND ----------

y_pred = model.predict(X_test)

# COMMAND ----------

# Test Partial Logistic Regression
report = classification_report(y_test, y_pred)
print(report)

# COMMAND ----------

plot_confusion_matrix(model, X_test, y_test)

# COMMAND ----------

# Test Total Logistic Regression
report = classification_report(y_test[test_features['RangoEdadMora'] == 0], y_pred[test_features['RangoEdadMora'] == 0])
print(report)

# COMMAND ----------

# Test Total Logistic Regression
report = classification_report(y_test[test_features['RangoEdadMora'].apply(lambda v: 0 < v <= 90)], y_pred[test_features['RangoEdadMora'].apply(lambda v: 0 < v <= 90)])
print(report)

# COMMAND ----------

# Test Total Logistic Regression
report = classification_report(y_test[test_features['RangoEdadMora'] > 90], y_pred[test_features['RangoEdadMora'] > 90])
print(report)

# COMMAND ----------

plot_confusion_matrix(model, X_test, y_test)

# COMMAND ----------

# Test Total Random Forest
report = classification_report(y_test[test_features['RangoEdadMora'] == 0], y_pred[test_features['RangoEdadMora'] == 0])
print(report)

# COMMAND ----------

# Test Total Random Forest
report = classification_report(y_test[test_features['RangoEdadMora'].apply(lambda v: 0 < v <= 90)], y_pred[test_features['RangoEdadMora'].apply(lambda v: 0 < v <= 90)])
print(report)

# COMMAND ----------

# Test Total Random Forest
report = classification_report(y_test[test_features['RangoEdadMora'] > 90], y_pred[test_features['RangoEdadMora'] > 90])
print(report)

# COMMAND ----------

plot_confusion_matrix(model, X_test, y_test)

# COMMAND ----------

# Test Total Random Forest 2
report = classification_report(y_test[test_features['RangoEdadMora'] == 0], y_pred[test_features['RangoEdadMora'] == 0])
print(report)

# COMMAND ----------

# Test Total Random Forest 2
report = classification_report(y_test[test_features['RangoEdadMora'].apply(lambda v: 0 < v <= 90)], y_pred[test_features['RangoEdadMora'].apply(lambda v: 0 < v <= 90)])
print(report)

# COMMAND ----------

# Test Total Random Forest 2
report = classification_report(y_test[test_features['RangoEdadMora'] > 90], y_pred[test_features['RangoEdadMora'] > 90])
print(report)

# COMMAND ----------

plot_confusion_matrix(model, X_test[test_features['RangoEdadMora'] > 90], y_test[test_features['RangoEdadMora'] > 90])

# COMMAND ----------

train_features[train_features['RangoEdadMora'] == 0].groupby(['Pago']).count()

# COMMAND ----------

# Test Total Random Forest 3
report = classification_report(y_test[test_features['RangoEdadMora'] == 0], y_pred[test_features['RangoEdadMora'] == 0])
print(report)

# COMMAND ----------

# Test Total Random Forest 3
report = classification_report(y_test[test_features['RangoEdadMora'].apply(lambda v: 0 < v <= 90)], y_pred[test_features['RangoEdadMora'].apply(lambda v: 0 < v <= 90)])
print(report)

# COMMAND ----------

# Test Total Random Forest 3
report = classification_report(y_test[test_features['RangoEdadMora'] > 90], y_pred[test_features['RangoEdadMora'] > 90])
print(report)

# COMMAND ----------

plot_confusion_matrix(model, X_test[test_features['RangoEdadMora'] > 90], y_test[test_features['RangoEdadMora'] > 90])

# COMMAND ----------

y_proba = model.predict_proba(X_test)[:, 1]

# COMMAND ----------

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# COMMAND ----------

cal_y, cal_x = calibration_curve(y_test, y_proba, n_bins = 20)

# COMMAND ----------

fig, ax = plt.subplots()
# only these two lines are calibration curves
plt.plot(cal_x, cal_y, marker = 'o', linewidth = 1)

# reference line, legends, and axis labels
line = mlines.Line2D([0, 1], [0, 1], color = 'black')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
fig.suptitle('Calibration plot')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('True probability in each bin')
plt.show()

# COMMAND ----------


