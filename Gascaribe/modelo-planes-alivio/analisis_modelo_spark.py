# Databricks notebook source
import os
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import when
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler

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
query = 'SELECT * FROM MPAlivios.Datos WHERE Entrenamiento = ' + ('1' if is_training else '0')

df = spark.read \
  .format("com.databricks.spark.sqldw") \
  .option("url", sqlDwUrl) \
  .option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
  .option("forwardSparkAzureStorageCredentials", "true") \
  .option("maxStrLength", "1024" ) \
  .option("query", query) \
  .load()

# COMMAND ----------

# Data Preparation
df = df.filter('Categoria = 1')

# Select columns
df = df.drop('FechaCierre', 'Entrenamiento', 'Año', 'Mes', 'Dia', 'Categoria', \
  'Barrio', 'Departamento', 'FinanciacionMayo', \
  'FinanciacionJunio', 'FinanciacionJulio', 'RecaudoMayo', \
  'RecaudoJunio', 'RecaudoJulio', 'RangoEdadMoraFinal', 'MoraMaxima')

if is_training:
  df = df.drop('Contrato')
  
# Class condensation
df = df.withColumn('RangoEdadMora', when(df.RangoEdadMora > 480, 480).otherwise(df.RangoEdadMora))

# Outlier Removal
if is_training:
  df = df.filter('DeudaCorrienteNoVencidaGas <= 2.5 * 1e6')
  df = df.filter('DeudaCorrienteVencidaGas <= 5 * 1e6')
  df = df.filter('DeudaDiferidaGas <= 20 * 1e6')
  df = df.filter('DeudaCorrienteNoVencidaBrilla <= 1.2 * 1e6')
  df = df.filter('DeudaCorrienteVencidaBrilla <= 3 * 1e6')
  df = df.filter('DeudaDiferidaBrilla <= 10 * 1e6')
  df = df.filter('DeudaCorrienteNoVencidaOtros <= 0.1 * 1e6')
  df = df.filter('DeudaCorrienteVencidaOtros <= 0.5 * 1e6')
  df = df.filter('Cuota <= 1.5 * 1e6')

# Output variable
if is_training:
  df = df.withColumn('label', when(df.EdadMora == 0, when(df.EdadMoraFinal == 0, 1).otherwise(0)) \
    .otherwise(when(df.EdadMoraFinal - df.EdadMora < 28, 1).otherwise(0)))

df = df.drop('EdadMora', 'EdadMoraFinal')

# COMMAND ----------

train_df, test_df = df.randomSplit([0.8,  0.2])

# COMMAND ----------

# Data preparation pipeline
indexer = StringIndexer(inputCols = ['Localidad', 'Refinanciado'], outputCols = ['Localidad_ind', 'Refinanciado_ind'])
encoder = OneHotEncoder(inputCols = ['Localidad_ind', 'Subcategoria', 'RangoEdadMora', 'Refinanciado_ind'], outputCols = ['Localidad_enc', 'Subcategoria_enc', 'RangoEdadMora_enc', 'Refinanciado_enc'])
assembler = VectorAssembler(inputCols = ['DeudaCorrienteNoVencidaGas', 'DeudaCorrienteVencidaGas', 'DeudaDiferidaGas', 'DeudaCorrienteNoVencidaBrilla', \
                                        'DeudaCorrienteVencidaBrilla', 'DeudaDiferidaBrilla', 'DeudaCorrienteNoVencidaOtros', \
                                        'DeudaCorrienteVencidaOtros', 'DeudaDiferidaOtros', 'CantRefiUltimoAño', 'CantHistoriaRefi', \
                                        'Veces30Gas', 'Veces60Gas', 'Veces90Gas', 'VecesMas90Gas', 'Veces30Brilla', \
                                        'Veces60Brilla', 'Veces90Brilla', 'VecesMas90Brilla', 'Veces30Otros', 'Veces60Otros', \
                                        'Veces90Otros', 'VecesMas90Otros', 'Suspensiones', 'Reconexiones', 'Cuota', 'PlazoGas', \
                                        'CuotasPendientesGas', 'SaldoInicialGas', 'PlazoBrilla', 'CuotasPendientesBrilla', \
                                        'SaldoInicialBrilla', 'Localidad_enc', 'Subcategoria_enc', 'RangoEdadMora_enc', 'Refinanciado_enc'], outputCol = "assembledFeatures")
scaler = StandardScaler(inputCol = 'assembledFeatures', outputCol = 'features')
pipeline = Pipeline(stages=[indexer, encoder, assembler, scaler])
preparation = pipeline.fit(train_df)

# COMMAND ----------

# Oversampling
df_a = train_df.filter(train_df['label'] == 0)
df_b = train_df.filter(train_df['label'] == 1)
a_len = df_a.count()
b_len = df_b.count()
if a_len < b_len:
  df_a, df_b, a_len, b_len = df_b, df_a, b_len, a_len
ratio = a_len / b_len
df_b = df_b.sample(withReplacement = True, fraction = ratio)
train_df = df_a.unionAll(df_b)

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

classifier = LogisticRegression()

paramGrid = ParamGridBuilder() \
    .addGrid(classifier.maxIter, [100, 500, 1000]) \
    .addGrid(classifier.elasticNetParam , [0.01, 0.05, 0.1]) \
    .build()

crossval = CrossValidator(estimator = classifier,
                          estimatorParamMaps = paramGrid,
                          evaluator = BinaryClassificationEvaluator(),
                          numFolds = 4)

# COMMAND ----------

X = preparation.transform(train_df)
cv_fit = crossval.fit(X)

# COMMAND ----------

print(cv_fit.bestModel.getMaxIter(), cv_fit.bestModel.getElasticNetParam())

# COMMAND ----------

X_test = preparation.transform(test_df)
X_pred = cv_fit.bestModel.transform(X_test)
res_df = X_pred[['RangoEdadMora', 'label', 'prediction']].toPandas()
x_rem, y_real, y_pred = res_df[['RangoEdadMora']], res_df['label'], res_df['prediction']

# COMMAND ----------

from sklearn.metrics import classification_report

# COMMAND ----------

idx_list = x_rem.apply(lambda x: 1 if x['RangoEdadMora'] == 0 else 0, axis = 1) == 1
report = classification_report(y_real[idx_list], y_pred[idx_list])
print(report)

# COMMAND ----------

idx_list = x_rem.apply(lambda x: 1 if x['RangoEdadMora'] <= 90 and x['RangoEdadMora'] > 0 else 0, axis = 1) == 1
report = classification_report(y_real[idx_list], y_pred[idx_list])
print(report)

# COMMAND ----------

idx_list = x_rem.apply(lambda x: 1 if x['RangoEdadMora'] > 90 else 0, axis = 1) == 1
report = classification_report(y_real[idx_list], y_pred[idx_list])
print(report)

# COMMAND ----------

classifier = RandomForestClassifier()

paramGrid = ParamGridBuilder() \
    .addGrid(classifier.maxDepth, [3, 5, 8]) \
    .addGrid(classifier.numTrees, [10, 50, 100]) \
    .build()

crossval = CrossValidator(estimator = classifier,
                          estimatorParamMaps = paramGrid,
                          evaluator = BinaryClassificationEvaluator(),
                          numFolds = 4)

# COMMAND ----------

X = preparation.transform(train_df)
cv_fit = crossval.fit(X)

# COMMAND ----------

print(cv_fit.bestModel.getMaxDepth(), cv_fit.bestModel.getNumTrees)

# COMMAND ----------

from sklearn.metrics import classification_report

# COMMAND ----------

classifier = RandomForestClassifier(numTrees = 100, maxDepth = 10)
X_train = preparation.transform(train_df)
cl_fit = classifier.fit(X_train)

# COMMAND ----------

X_test = preparation.transform(test_df)
X_pred = cl_fit.transform(X_test)
res_df = X_pred[['RangoEdadMora', 'label', 'prediction']].toPandas()

# COMMAND ----------

x_rem, y_real, y_pred = res_df[['RangoEdadMora']], res_df['label'], res_df['prediction']

# COMMAND ----------

idx_list = x_rem.apply(lambda x: 1 if x['RangoEdadMora'] == 0 else 0, axis = 1) == 1
report = classification_report(y_real[idx_list], y_pred[idx_list])
print(report)

# COMMAND ----------

idx_list = x_rem.apply(lambda x: 1 if x['RangoEdadMora'] <= 90 and x['RangoEdadMora'] > 0 else 0, axis = 1) == 1
report = classification_report(y_real[idx_list], y_pred[idx_list])
print(report)

# COMMAND ----------

idx_list = x_rem.apply(lambda x: 1 if x['RangoEdadMora'] > 90 else 0, axis = 1) == 1
report = classification_report(y_real[idx_list], y_pred[idx_list])
print(report)

# COMMAND ----------

classifier = GBTClassifier(maxIter = 50, maxDepth = 8)
X_train = preparation.transform(train_df)
cl_fit = classifier.fit(X_train)

# COMMAND ----------

X_test = preparation.transform(test_df)
X_pred = cl_fit.transform(X_test)
res_df = X_pred[['RangoEdadMora', 'label', 'prediction']].toPandas()

# COMMAND ----------

x_rem, y_real, y_pred = res_df[['RangoEdadMora']], res_df['label'], res_df['prediction']

# COMMAND ----------

idx_list = x_rem.apply(lambda x: 1 if x['RangoEdadMora'] == 0 else 0, axis = 1) == 1
report = classification_report(y_real[idx_list], y_pred[idx_list])
print(report)

# COMMAND ----------

idx_list = x_rem.apply(lambda x: 1 if x['RangoEdadMora'] <= 90 and x['RangoEdadMora'] > 0 else 0, axis = 1) == 1
report = classification_report(y_real[idx_list], y_pred[idx_list])
print(report)

# COMMAND ----------

idx_list = x_rem.apply(lambda x: 1 if x['RangoEdadMora'] > 90 else 0, axis = 1) == 1
report = classification_report(y_real[idx_list], y_pred[idx_list])
print(report)

# COMMAND ----------


