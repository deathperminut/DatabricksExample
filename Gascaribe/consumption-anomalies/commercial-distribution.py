# Databricks notebook source
import pandas as pd
import scipy.stats as stats
from scipy.stats import kstest
import matplotlib.pyplot as plt

# COMMAND ----------

# Fetch the data from the table analiticagdc.desviacionconsumos.factconsumo into a pandas dataframe
data = spark.sql("SELECT CAST(ROUND(VolumenNormalizado) AS INT) AS Volumen FROM analiticagdc.desviacionconsumos.factconsumo WHERE FechaPeriodo = '2024-03-01' AND IdCategoria = 2 AND VolumenNormalizado <= 1000 AND VolumenNormalizado IS NOT NULL AND VolumenNormalizado > 50").toPandas()

# COMMAND ----------

display(data)

# COMMAND ----------

import numpy as np
from scipy import stats

# Fit the sample data to various distributions
gamma_params = stats.gamma.fit(data)
norm_params = stats.lognorm.fit(data)
exp_params = stats.expon.fit(data)

# Calculate the AIC score for each distribution
gamma_aic = stats.gamma.nnlf(gamma_params, data)
norm_aic = stats.lognorm.nnlf(norm_params, data)
exp_aic = stats.expon.nnlf(exp_params, data)

# Determine the distribution with the lowest AIC score
if gamma_aic < norm_aic and gamma_aic < exp_aic:
    best_dist = 'gamma'
    best_params = gamma_params
elif norm_aic < gamma_aic and norm_aic < exp_aic:
    best_dist = 'lognorm'
    best_params = norm_params
else:
    best_dist = 'expon'
    best_params = exp_params

best_dist, best_params

# COMMAND ----------

# MAGIC %md
# MAGIC Se detecta que el consumo reside se ajusta a una distribución log-normal
# MAGIC Esta conclusión se obtuvo utilizando una prueba de bondad de ajuste y posteriormente el criterio de Akaike (AIC) para escoger la distribución más cercana entre las tres candidatas (Gamma, LogNormal y Exponencial). Para el caso de los usuarios comerciales, se tomó una variación únicamente de una deviación estandar.

# COMMAND ----------

print("moda:", data.mode())
print("media:", data.mean())
print("desviacion:", data.std())

# COMMAND ----------

print("Rango limite superior valor aceptado:", (data.mode().iloc[0][0] + 1 * data.std().iloc[0]), ' - ', (data.mean().iloc[0] + 1 * data.std().iloc[0]))

# COMMAND ----------

# MAGIC %md
