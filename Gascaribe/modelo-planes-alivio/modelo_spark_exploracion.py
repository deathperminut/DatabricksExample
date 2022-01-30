# Databricks notebook source
import os
from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql.functions import when, udf
from pyspark.ml.classification import GBTClassifier, GBTClassificationModel
from pyspark.ml import Pipeline, PipelineModel
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
is_training = False

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

# Load Model
preparation = PipelineModel.load('/app/MPAlivios_Pipeline')
cl_fit = GBTClassificationModel.load('/app/MPAlivios_Classifier')
# Make predictions
X_inp = preparation.transform(df)
X_pred = cl_fit.transform(X_inp)
# Tranform predictions
extract_probability = udf(lambda v: float(v[1]), FloatType())
X_res = X_pred.withColumn('ProbabilidadPago', extract_probability('probability'))
X_res = X_res[['Contrato', 'ProbabilidadPago']]
# Save predictions
X_res.write \
  .format("com.databricks.spark.sqldw") \
  .option("url", sqlDwUrl) \
  .option("forwardSparkAzureStorageCredentials", "true") \
  .option("dbTable", "MPAlivios.Resultados") \
  .option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
  .mode("overwrite") \
  .save()

# COMMAND ----------


