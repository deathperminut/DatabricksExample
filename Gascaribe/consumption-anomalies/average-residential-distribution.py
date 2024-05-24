# Databricks notebook source
import pandas as pd
from scipy.stats import kstest
from pyspark.sql.functions import col
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns

# COMMAND ----------

# Fetch the data from the table analiticagdc.desviacionconsumos.factconsumo into a pandas dataframe
data = spark.sql(
    """
        SELECT 
            ROUND((fc.VolumenNormalizado - fcp.Volumen) / fcp.Volumen, 2) AS Variacion 
        FROM analiticagdc.desviacionconsumos.factconsumo fc
        INNER JOIN analiticagdc.desviacionconsumos.factpromedioconsumo fcp
            ON fc.IdProducto = fcp.IdProducto
            AND fcp.Volumen > 0
        WHERE fc.FechaPeriodo = '2024-03-01' 
        AND fc.IdCategoria = 1 
        AND fc.VolumenNormalizado > 0 
        AND fc.VolumenNormalizado IS NOT NULL
    """
)\
.filter(col("Variacion") < 5)\
.toPandas()

# COMMAND ----------

display(data)

# COMMAND ----------

# MAGIC %md
# MAGIC Se detecta que el consumo reside se ajusta a una distribución normal con apoyo del Teorema del Límite Central
# MAGIC
# MAGIC El Teorema del Límite Central es un concepto fundamental en estadística. Este teorema establece que si se toma una muestra suficientemente grande de una población, independientemente de la forma de la distribución de la población, la distribución de la media de la muestra se aproximará a una distribución normal. Esta aproximación mejora a medida que el tamaño de la muestra aumenta. (N > 900 000 para este analisis)
# MAGIC
# MAGIC El teorema se aplica cuando se tienen muchas observaciones independientes de una misma variable aleatoria (en este caso, el consumo).
# MAGIC
# MAGIC Es importante notar que el Teorema del Límite Central se aplica a la distribución de las medias de las muestras, no a la distribución de los datos individuales. (En este caso se aplica a la variación del consumo de cada usuario)

# COMMAND ----------

print("media:", data.mean())
print("desviacion:", data.std())

# COMMAND ----------

print("Rango valor aceptado:", (data.mean().iloc[0] - 2 * data.std().iloc[0]), ' - ', (data.mean().iloc[0] + 2 * data.std().iloc[0]))

# COMMAND ----------

# MAGIC %md
