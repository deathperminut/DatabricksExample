# Databricks notebook source
from scipy.stats import anderson
import pandas as pd

# COMMAND ----------

# Fetch the data from the table analiticagdc.desviacionconsumos.factconsumo into a pandas dataframe
data = spark.sql("SELECT ROUND(VolumenNormalizado) AS Volumen FROM analiticagdc.desviacionconsumos.factconsumo WHERE FechaPeriodo = '2023-01-01' AND IdCategoria = 1 AND VolumenNormalizado > 0 AND VolumenNormalizado IS NOT NULL").toPandas()

# COMMAND ----------

display(data)

# COMMAND ----------

# Perform Anderson-Darling test on the data
result = anderson(data['Volumen'], dist='norm')

# COMMAND ----------

# Get the test statistic and critical values
test_statistic = result.statistic
critical_values = result.critical_values

# COMMAND ----------

# Print the test statistic and critical values
print("Anderson-Darling Test Statistic:", test_statistic)
print("Critical Values for significance levels:", critical_values)
