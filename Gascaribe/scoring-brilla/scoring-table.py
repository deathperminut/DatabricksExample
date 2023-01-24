# Databricks notebook source
datalake = 'gdcbidatalake.dfs.core.windows.net'
goldScoringFNB = 'abfss://gold@'+datalake+'/scoringFNB'

spark.conf.set('fs.azure.account.key.'+datalake, dbutils.secrets.get(scope="gdcbi", key="datalakekey"));

spark.sql("""
CREATE TABLE IF NOT EXISTS {goldScoringFNB} ( 
    IdContrato      BIGINT,
    CupoAsignado    BIGINT,
    Identificacion  VARCHAR(20),
    Tipo            VARCHAR(50),
    Nodo            INTEGER,
    Riesgo          VARCHAR(20),
    Categoria       INTEGER,
    Estrato         INTEGER,
    FechaPrediccion DATE
    )
""")

# COMMAND ----------

datalake = "gdcbidatalake.dfs.core.windows.net";

spark.conf.set(s"fs.azure.account.key.$datalake", dbutils.secrets.get(scope="gdcbi", key="datalakekey"));

goldScoringFNB = s"delta.`abfss://gold@$datalake/scoringFNB`"

# COMMAND ----------

count = spark.sql(s"""
SELECT * FROM $goldScoringFNB
""") 

display(count)

# COMMAND ----------


