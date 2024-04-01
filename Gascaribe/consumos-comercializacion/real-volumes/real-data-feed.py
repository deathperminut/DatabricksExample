# Databricks notebook source
from delta.tables import *
from pyspark.sql.functions import asc, desc, col
from pyspark.sql.functions import current_timestamp, date_sub

# COMMAND ----------

user = dbutils.secrets.get(scope='gascaribe', key='com-user')
password = dbutils.secrets.get(scope='gascaribe', key='com-password')
host = dbutils.secrets.get(scope='gascaribe', key='com-host')
port = dbutils.secrets.get(scope='gascaribe', key='com-port')
database = dbutils.secrets.get(scope='gascaribe', key='com-database')

# COMMAND ----------

results = DeltaTable.forName(spark, 'analiticagdc.comercializacion.factvolumen').toDF()\
    .select(col("idcomercializacion").alias("id"), col("volumen").alias("volumen_real"),col("fecha"))\
    .filter(col("fecha") >= date_sub(current_timestamp(), 180))\
    .filter(col("fecha") >= "2024-01-01")

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
    table="public.volumenes_bi", 
    mode="overwrite", 
    properties=properties
)
