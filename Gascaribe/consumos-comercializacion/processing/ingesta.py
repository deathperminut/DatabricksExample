# Databricks notebook source
import os
import pandas as pd
import numpy as np
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import *
from delta.tables import *

import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

# COMMAND ----------

spark.sql("""
CREATE TABLE IF NOT EXISTS analiticagdc.comercializacion.ingesta (
  IdComercializacion int,
  Estacion           varchar(100),
  TipoUsuario        varchar(50),
  IdDispositivo      varchar(15),
  Fecha              date,
  Volumen            float
)
"""
)

# COMMAND ----------

volumenes = DeltaTable.forName(spark, 'analiticagdc.comercializacion.factvolumen').toDF()
estaciones = DeltaTable.forName(spark, 'production.comercializacion.estaciones').toDF()

# COMMAND ----------

ingesta = volumenes.alias("v") \
.join(estaciones.alias("e"), col("v.IdComercializacion") == col("e.Id"), 'left') \
.selectExpr( "v.IdComercializacion",
             "descripcion as Estacion",
             "tipo_usuario as TipoUsuario",
             "id_electrocorrector as IdDispositivo",
             "Fecha",
             "Volumen")

# COMMAND ----------

deltaTable_ingesta = DeltaTable.forName(spark, 'analiticagdc.comercializacion.ingesta')

mapping =  {
              "IdComercializacion"  : "df.IdComercializacion",
              "Estacion"            : "df.Estacion",
              "TipoUsuario"         : "df.TipoUsuario",
              "IdDispositivo"       : "df.IdDispositivo",
              "Fecha"               : "df.Fecha",
              "Volumen"             : "df.Volumen"
    }

deltaTable_ingesta.alias('t') \
  .merge( ingesta.alias('df'), 't.IdComercializacion = df.IdComercializacion AND t.Fecha = df.Fecha') \
  .whenMatchedUpdate(set=mapping) \
  .whenNotMatchedInsert(values=mapping) \
  .execute()

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT * 
# MAGIC FROM analiticagdc.comercializacion.ingesta
# MAGIC WHERE IdComercializacion = 166
