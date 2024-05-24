# Databricks notebook source
# DBTITLE 1,Importar Librerias
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import *
from delta.tables import *

# COMMAND ----------

# DBTITLE 1,Creacion de tabla
spark.sql("""
CREATE OR REPLACE TABLE analiticagdc.desviacionconsumos.factpromedioconsumo (
    IdProducto     BIGINT,
    Volumen        DOUBLE
)
"""
)

# COMMAND ----------

# DBTITLE 1,Preparacion de tablas base
factconsumo = DeltaTable.forName(spark, 'analiticagdc.desviacionconsumos.factconsumo').toDF()\
    .filter(col("FechaPeriodo") >= '2023-09-01')\
        .filter(col("FechaPeriodo") <= '2024-02-01')\
    .select(
        col("IdProducto"),
        col("VolumenNormalizado")
    )

# COMMAND ----------

# DBTITLE 1,Transformaciones
factpromedioconsumo = factconsumo\
    .groupBy("IdProducto")\
    .agg(avg("VolumenNormalizado").cast(DoubleType()).alias("Volumen"))

# COMMAND ----------

# DBTITLE 1,Escribir en tabla final
factpromedioconsumo.write.mode("overwrite").saveAsTable("analiticagdc.desviacionconsumos.factpromedioconsumo")
