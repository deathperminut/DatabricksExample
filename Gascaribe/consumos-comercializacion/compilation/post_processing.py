# Databricks notebook source
import os
import pandas as pd
import numpy as np
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import *     
from delta.tables import *
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import itertools
import math
import random
import time
from datetime import date,datetime,timedelta,timezone
from scipy.optimize import minimize
from pandas.api.indexers import BaseIndexer

import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS analiticagdc.comercializacion.forecastresults (
# MAGIC   IdComercializacion       int,
# MAGIC   Estacion                 varchar(100),
# MAGIC   IdDispositivo            varchar(15),
# MAGIC   TipoUsuario              varchar(50),
# MAGIC   Modelo                   string,
# MAGIC   Fecha                    date,
# MAGIC   Volumen                  float,
# MAGIC   Prediccion               float,
# MAGIC   Error                    float
# MAGIC )

# COMMAND ----------

mediapredictions = DeltaTable.forName(spark, 'analiticagdc.comercializacion.mediapredictions').toDF()
prophetpredictions_ = DeltaTable.forName(spark, 'analiticagdc.comercializacion.prophetpredictions').toDF()
real_values = DeltaTable.forName(spark, 'analiticagdc.comercializacion.ingesta').toDF()

estaciones_media = DeltaTable.forName(spark, 'analiticagdc.comercializacion.mediapredictions').toDF() \
    .selectExpr( 'IdComercializacion' ).drop_duplicates()


prophetpredictions = prophetpredictions_.filter( ~(col("IdComercializacion").isin( estaciones_media.rdd.flatMap(lambda x: x).collect() ) ) )

predictions = prophetpredictions.union(mediapredictions)

results =  predictions.alias("p") \
    .join( real_values.alias("rv"), (col("p.IdComercializacion") == col("rv.IdComercializacion")) & (col("p.Fecha") == col("rv.Fecha")), 'left'  ) \
    .withColumn( 'Error', abs( col("p.Prediccion") - col("rv.Volumen") )/col("rv.Volumen") )  \
    .selectExpr( 'p.IdComercializacion',
                 'rv.Estacion',
                 'rv.IdDispositivo',
                 'rv.TipoUsuario',
                 'p.Modelo',
                 'p.Fecha',
                 'rv.Volumen',
                 'p.Prediccion',
                 'Error'  )


# COMMAND ----------

deltaTable_forecastresults = DeltaTable.forName(spark, 'analiticagdc.comercializacion.forecastresults')

mapping =  {
              "IdComercializacion" :  "df.IdComercializacion",
              "Estacion"           :  "df.Estacion",
              "IdDispositivo"      :  "df.IdDispositivo",
              "TipoUsuario"        :  "df.TipoUsuario",
              "Modelo"             :  "df.Modelo",
              "Fecha"              :  "df.Fecha",
              "Volumen"            :  "df.Volumen",
              "Prediccion"         :  "df.Prediccion",
              "Error"              :  "df.Error"
    }

deltaTable_forecastresults.alias('t') \
  .merge( results.alias('df'), 't.IdComercializacion = df.IdComercializacion AND t.Fecha = df.Fecha') \
  .whenMatchedUpdate(set=mapping) \
  .whenNotMatchedInsert(values=mapping) \
  .execute()
