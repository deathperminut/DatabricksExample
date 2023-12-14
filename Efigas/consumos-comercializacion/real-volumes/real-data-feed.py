# Databricks notebook source
import psycopg2
from delta.tables import *
from pyspark.sql.functions import asc, desc

# COMMAND ----------

# Constants
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

# Data Ingestion
query = """
    SELECT *
    FROM ComercializacionML.IngestaBricks  
    """

deltaDF = spark.read \
  .format("com.databricks.spark.sqldw") \
  .option("url", sqlDwUrl) \
  .option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
  .option("forwardSparkAzureStorageCredentials", "true") \
  .option("maxStrLength", "1024" ) \
  .option("query", query) \
  .load()

# COMMAND ----------

deltaDF.write.mode("overwrite").saveAsTable("analiticaefg.comercializacion.factvolumenreal")

# COMMAND ----------

user = dbutils.secrets.get(scope='efigas', key='com-user')
password = dbutils.secrets.get(scope='efigas', key='com-password')
host = dbutils.secrets.get(scope='efigas', key='com-host')
port = dbutils.secrets.get(scope='efigas', key='com-port')
database = dbutils.secrets.get(scope='efigas', key='com-database')

# COMMAND ----------

results = DeltaTable.forName(spark, 'analiticaefg.comercializacion.factvolumenreal').toDF()\
    .select("idestacion", "fecha", "volumen")

# COMMAND ----------

# Define the connection properties
url = f"jdbc:postgresql://{host}:{port}/{database}"
properties = {
    "driver": "org.postgresql.Driver",
    "user": user,
    "password": password
}

# COMMAND ----------

results.write.jdbc(
    url, 
    table="public.volumenes_bi", 
    mode="overwrite", 
    properties=properties
)
