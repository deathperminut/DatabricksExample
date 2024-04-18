# Databricks notebook source
import pandas as pd
import scipy.stats as stats
from scipy.stats import kstest

# COMMAND ----------

# Fetch the data from the table analiticagdc.desviacionconsumos.factconsumo into a pandas dataframe
data = spark.sql("SELECT CAST(ROUND(VolumenNormalizado) AS INT) AS Volumen FROM analiticagdc.desviacionconsumos.factconsumo WHERE FechaPeriodo = '2023-01-01' AND IdCategoria = 1 AND VolumenNormalizado > 0 AND VolumenNormalizado IS NOT NULL").toPandas()

# COMMAND ----------

display(data)

# COMMAND ----------

for i in range (1, 20):
    for j in range (1, 20):
        gof = kstest(data['Volumen'], 'f', args=(i, j))
        if(gof[1] > 0.05):
            print(gof, 'dfn=', i, ' dfd=', j)
