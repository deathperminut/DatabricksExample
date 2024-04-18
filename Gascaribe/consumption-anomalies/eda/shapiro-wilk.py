# Databricks notebook source
from scipy.stats import shapiro
import pandas as pd

# COMMAND ----------

# Fetch the data from the table analiticagdc.desviacionconsumos.factconsumo into a pandas dataframe
data = spark.sql("SELECT ROUND(VolumenNormalizado) AS Volumen FROM analiticagdc.desviacionconsumos.factconsumo WHERE FechaPeriodo = '2023-01-01' AND IdCategoria = 1 AND VolumenNormalizado > 0 AND VolumenNormalizado IS NOT NULL").toPandas()

# COMMAND ----------

display(data)

# COMMAND ----------

# Perform the Shapiro-Wilk test
stat, p_value = shapiro(data['Volumen'])

# COMMAND ----------

# Check if the data is normally distributed based on the p-value
is_normal = True if p_value > 0.05 else False

# Print the result
print("Is the data normally distributed? ", is_normal, "pvalue:", p_value)
