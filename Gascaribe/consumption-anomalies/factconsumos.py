# Databricks notebook source
# DBTITLE 1,Importar Librerias
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import *
from delta.tables import *
from pyspark.sql.window import Window

# COMMAND ----------

# DBTITLE 1,Creacion de tabla
spark.sql("""
CREATE TABLE IF NOT EXISTS analiticagdc.desviacionconsumos.factconsumo (
    IdProducto     BIGINT,
    IdCategoria    SMALLINT,
    IdSubcategoria SMALLINT,
    FechaPeriodo   DATE,
    Volumen        INT
)
"""
)

# COMMAND ----------

# DBTITLE 1,Preparacion de tablas base
prioridadDias = Window.partitionBy(col("IdProducto"), col("IdPeriodoFactura")).orderBy(col("IdMetodoPrioridad"), col("FechaConsumo").desc())
prioridadMetodo = Window.partitionBy(col("IdProducto"), col("IdPeriodoFactura")).orderBy(col("FechaConsumoPrioridad").desc(), col("IdMetodoPrioridad").desc())

calculoconsumo = DeltaTable.forName(spark, 'bigdc.facturacion.factcalculoconsumo').toDF()\
    .filter(col("IdMetodoCalculoConsumo") != 4)\
    .select(
        col("IdProducto").cast(LongType()),
        col("IdMetodoCalculoConsumo"),
        col("Consumo"),
        col("IdPeriodoFactura"),
        col("FechaConsumo"),
        col("DiasConsumo")
    )\
    .withColumn(
        "IdMetodoPrioridad",
        when(
            col("IdMetodoCalculoConsumo") == 2,
            8
        )
        .otherwise("IdMetodoCalculoConsumo")
    )\
    .withColumn(
        "FechaConsumoPrioridad",
        when(
            col("IdMetodoCalculoConsumo") == 2,
            lit("1900-01-01").cast(DateType())
        )
        .otherwise("FechaConsumo").cast(DateType())
    )\
    .withColumn(
        "PrioridadDiasConsumo",
        row_number().over(prioridadDias)
    )\
    .withColumn(
        "PrioridadMetodo",
        row_number().over(prioridadDias)
    )\
    .groupBy(
        "IdProducto",
        "IdPeriodoFactura"
    )\
    .agg(
        sum(col("Consumo")).cast(DecimalType(22, 2)).alias("Unidades"),
        min(col("IdMetodoCalculoConsumo")).alias("MinimoMetodo"),
        max(
            when(
                col("PrioridadMetodo") == 1,
                col("IdMetodoCalculoConsumo")
            ).otherwise(0)
        ).alias("IdMetodoCalculoConsumo"),
        max(
            when(
                col("PrioridadDiasConsumo") == 1,
                col("DiasConsumo")
            ).otherwise(0)
        ).alias("DiasConsumo")
    )

periodofactura = DeltaTable.forName(spark, 'bigdc.facturacion.dimperiodofactura').toDF()\
    .filter(col("is_current") == True)\
    .filter(col("FechaMesPeriodo") >= '2022-01-01')\
    .select(
        col("IdPeriodoFactura"),
        col("FechaMesPeriodo").alias("FechaPeriodo")
    )\
    .withColumn(
        "FechaCierre",
        last_day("FechaPeriodo")
    )

cierres = DeltaTable.forName(spark, 'bigdc.cartera.factresumencierredia').toDF()\
    .filter(col("IdCategoria") <= 2)\
    .filter(col("FechaCierre") == last_day(col("FechaCierre")))\
    .select(
        col("IdProducto").cast(LongType()),
        col("FechaCierre"),
        col("IdCategoria"),
        col("IdSubcategoria")
    )

# COMMAND ----------

# DBTITLE 1,Transformaciones
factconsumos = calculoconsumo.join(periodofactura, "IdPeriodoFactura")\
    .join(cierres, ["IdProducto", "FechaCierre"])\
    .drop("IdPeriodoFactura", "IdCuentaCobro")\
    .groupBy("IdProducto", "IdCategoria", "IdSubcategoria", "FechaPeriodo")\
    .agg(sum("Unidades").cast(IntegerType()).alias("Volumen"))

# COMMAND ----------

# DBTITLE 1,Escribir en tabla final
factconsumos.write.mode("overwrite").saveAsTable("analiticagdc.desviacionconsumos.factconsumo")
