# Databricks notebook source
model_name = "RFM_efg"

# COMMAND ----------

from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
import os
import mlflow
from databricks import feature_store
from datetime import date,datetime
from pyspark.sql import functions as F
from pyspark.sql.functions import sum,avg,max,count,col,lit



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

#%pip install -r $requirements_path

# COMMAND ----------

# redefining key variables here because %pip and %conda restarts the Python interpreter
model_name = "RFM_efg"
input_table_name = "hive_metastore.default.efg_segmentacion_features"
inactive_table_name = "hive_metastore.default.EFG_inactivos_brilla"
output_table_path = "/FileStore/batch-inference/RFM_efg"

# COMMAND ----------

# load table as a Spark DataFrame
table = spark.table(input_table_name)
fs = feature_store.FeatureStoreClient()
# optionally, perform additional data processing (may be necessary to conform the schema)

# COMMAND ----------

import mlflow

model_uri = f"models:/{model_name}/Production"

# create spark user-defined function for model prediction
preds = fs.score_batch(model_uri, table, "cluster int, name string")

# COMMAND ----------

X = preds.withColumn("cluster", preds["prediction"]["cluster"]).withColumn("name", preds["prediction"]["name"]).drop("prediction")

# COMMAND ----------

X.show()

# COMMAND ----------

inactivos = spark.table(inactive_table_name)
newX = X[['Identificacion', 'Recency', 'Frequency', 'Monetary', 'cluster', 'Score', "name"]].union(inactivos[['Identificacion', 'Recency', 'Frequency', 'Monetary', 'cluster', 'Score', 'name']])

# COMMAND ----------

inactivos.show()

# COMMAND ----------

results = newX[['Identificacion', 'cluster', 'Score','name', 'Recency', 'Frequency', 'Monetary']]
results = results.withColumn("FechaPrediccion", F.current_date())
results = results.withColumnRenamed("cluster", "Segmento").withColumnRenamed("name", "NombreSegmento").withColumnRenamed("Score", "Puntaje").withColumnRenamed("Recency", "Recencia").withColumnRenamed("Frequency", "Frecuencia").withColumnRenamed("Monetary", "Monetario")

results.write \
.format("com.databricks.spark.sqldw") \
.option("url", sqlDwUrl) \
.option("forwardSparkAzureStorageCredentials", "true") \
.option("dbTable", "ModeloRFMBrilla.StageSegmentosRFM") \
.option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
.mode("overwrite") \
.save()

# COMMAND ----------

results.show()

# COMMAND ----------

agrupados = newX.groupBy("name").agg(count("Recency").alias("CantidadDeUsuarios"), avg("Recency").alias("Recencia"), avg("Frequency").alias("Frecuencia"), avg("Monetary").alias("Monetario"))
group = agrupados.select(col("name").alias("Segmento"), col("CantidadDeUsuarios"), col("Recencia").cast("float"), col("Frecuencia").cast("float"), col("Monetario").cast("float"))

# COMMAND ----------

group.show()

# COMMAND ----------

group.write \
.format("com.databricks.spark.sqldw") \
.option("url", sqlDwUrl) \
.option("forwardSparkAzureStorageCredentials", "true") \
.option("dbTable", "ModeloRFMBrilla.ResumenSegmentos") \
.option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
.mode("overwrite") \
.save()

# COMMAND ----------

results = results.withColumn("is_current", lit(1))

# COMMAND ----------

results.write.mode('overwrite').saveAsTable('analiticaefg.brilla.segmentosrfm')
