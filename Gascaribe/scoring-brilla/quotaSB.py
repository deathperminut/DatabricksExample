# Databricks notebook source
import delta.tables
from delta.tables import *
from pyspark.sql.functions import asc, desc
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
from pyspark.sql.functions import col
from pyspark.sql.functions import current_timestamp

# COMMAND ----------

windowSpec  = Window.partitionBy("IdContrato").orderBy(desc("FechaPrediccion"))

results = DeltaTable.forName(spark, 'analiticagdc.brilla.quotaSB')\
    .toDF()\
    .withColumn("row_number", row_number().over(windowSpec))\
    .filter(col("row_number") == 1)\
    .select("IdContrato", "CupoAsignado", "Nodo", "Riesgo")\
    .withColumn("created_at", current_timestamp())\
    .withColumn("updated_at", current_timestamp())\

newColumns = ["contract_id", "assigned_quota", "nodo", "risk_level", "created_at", "updated_at"]

for i in range(len(results.columns)):
    results = results.withColumnRenamed(results.columns[i], newColumns[i])

# COMMAND ----------

user = dbutils.secrets.get(scope='gascaribe', key='sb-user')
password = dbutils.secrets.get(scope='gascaribe', key='sb-password')
host = dbutils.secrets.get(scope='gascaribe', key='sb-host')
port = dbutils.secrets.get(scope='gascaribe', key='sb-port')
database = dbutils.secrets.get(scope='gascaribe', key='sb-database')

# COMMAND ----------

# Define the connection properties
url = f"jdbc:postgresql://{host}:{port}/{database}"
properties = {
    "driver": "org.postgresql.Driver",
    "user": user,
    "password": password
}

# COMMAND ----------

results.write.option("truncate", "true").jdbc(
    url, 
    table="public.bi_assigned_quotas", 
    mode="overwrite", 
    properties=properties
)
