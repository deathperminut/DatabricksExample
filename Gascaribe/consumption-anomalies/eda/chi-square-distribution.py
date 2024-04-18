# Databricks notebook source
import scipy.stats as stats
import pandas as pd

# COMMAND ----------

# Fetch the data from the table analiticagdc.desviacionconsumos.factconsumo into a pandas dataframe
data = spark.sql("SELECT ROUND(VolumenNormalizado) AS Volumen FROM analiticagdc.desviacionconsumos.factconsumo WHERE FechaPeriodo = '2023-01-01' AND IdCategoria = 1 AND VolumenNormalizado > 0 AND VolumenNormalizado IS NOT NULL").toPandas()

# COMMAND ----------

display(data)

# COMMAND ----------

# I want to check if the Volume column follows a chi squared distribution

# Perform chi-square test for goodness of fit
chi2_stat, p_value = stats.chisquare(data['Volumen'], ddof=1)

# Check if p-value is less than 0.05
if p_value < 0.05:
    print("The Volume column does not follow a chi squared distribution.")
else:
    print("The Volume column follows a chi squared distribution.")
