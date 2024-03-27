# Databricks notebook source
from delta.tables import *
from pyspark.sql.functions import asc, desc, col

# COMMAND ----------

user = dbutils.secrets.get(scope='gascaribe', key='com-user')
password = dbutils.secrets.get(scope='gascaribe', key='com-password')
host = dbutils.secrets.get(scope='gascaribe', key='com-host')
port = dbutils.secrets.get(scope='gascaribe', key='com-port')
database = dbutils.secrets.get(scope='gascaribe', key='com-database')

# COMMAND ----------

results = DeltaTable.forName(spark, 'analiticagdc.comercializacion.factvolumen').toDF()\
    .select(col("idcomercializacion").alias("id"), col("volumen").alias("volumen_real"),col("fecha"))

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
