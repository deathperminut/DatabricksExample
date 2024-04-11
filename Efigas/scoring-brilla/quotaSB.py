# Databricks notebook source
import delta.tables
import psycopg2
from delta.tables import *
from pyspark.sql.functions import asc, desc
from pyspark.sql.functions import row_number
from pyspark.sql.functions import col
from pyspark.sql.functions import current_timestamp

# COMMAND ----------

results = DeltaTable.forName(spark, 'analiticaefg.brilla.quotaSB')\
    .toDF()\
    .dropDuplicates(["IdContrato"])\
    .select("IdContrato", "CupoAsignado", "Nodo", "Riesgo")\
    .withColumn("created_at", current_timestamp())\
    .withColumn("updated_at", current_timestamp())\

newColumns = ["contract_id", "assigned_quota", "nodo", "risk_level", "created_at", "updated_at"]

for i in range(len(results.columns)):
    results = results.withColumnRenamed(results.columns[i], newColumns[i])

# COMMAND ----------

user = dbutils.secrets.get(scope='efigas', key='sb-user')
password = dbutils.secrets.get(scope='efigas', key='sb-password')
host = dbutils.secrets.get(scope='efigas', key='sb-host')
port = dbutils.secrets.get(scope='efigas', key='sb-port')
database = dbutils.secrets.get(scope='efigas', key='sb-database')

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
