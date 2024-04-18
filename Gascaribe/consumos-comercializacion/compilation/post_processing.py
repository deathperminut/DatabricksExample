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

mediapredictions_ = DeltaTable.forName(spark, 'analiticagdc.comercializacion.mediapredictions').toDF()
prophetpredictions_ = DeltaTable.forName(spark, 'analiticagdc.comercializacion.prophetpredictions').toDF()
real_values = DeltaTable.forName(spark, 'analiticagdc.comercializacion.ingesta').toDF()
estaciones = DeltaTable.forName(spark, 'production.comercializacion.estaciones').toDF()
forecastresults_ = DeltaTable.forName(spark, 'analiticagdc.comercializacion.forecastresults').toDF()

fechas_tuneo = DeltaTable.forName(spark, 'analiticagdc.comercializacion.prophettunningparameter').toDF() \
.selectExpr( 'FechaRegistro' ).dropDuplicates().orderBy(desc(col('FechaRegistro')))
fecha_ultimo_tuneo_prophet = fechas_tuneo.collect()[0][0] 
try:
    fecha_penultimo_tuneo_prophet = fechas_tuneo.collect()[1][0]
except Exception as e:
    fecha_penultimo_tuneo_prophet = to_date(lit("1900-01-01")) 

estaciones_media_ayer = forecastresults_.filter( (col("Fecha") == fecha_penultimo_tuneo_prophet) & ( col("Modelo") == lit("Media") )  ).select( col("IdComercializacion") )
estaciones_prophet_ayer = forecastresults_.filter( (col("Fecha") == fecha_penultimo_tuneo_prophet) & ( col("Modelo") == lit("Prophet") ) ).select( col("IdComercializacion") )


media_ayer = mediapredictions_.filter( (col("IdComercializacion").isin( estaciones_media_ayer.rdd.flatMap(lambda x: x).collect() ) ) & ( col("Fecha") == fecha_penultimo_tuneo_prophet ) )
prophet_ayer = prophetpredictions_.filter( (col("IdComercializacion").isin( estaciones_prophet_ayer.rdd.flatMap(lambda x: x).collect() ) ) & ( col("Fecha") == fecha_penultimo_tuneo_prophet ) )
predictions_ayer = prophet_ayer.union(media_ayer)


estaciones_mal_portadas_1 = DeltaTable.forName(spark, 'analiticagdc.comercializacion.prophettunningparameter_06').toDF() \
    .filter( (col("Metric2") < 0.6) & ( (col("is_current") == 1) ) ).selectExpr( 'IdComercializacion' )

estaciones_mal_portadas_2_ = DeltaTable.forName(spark, 'analiticagdc.comercializacion.prophettunningparameter_06').toDF() \
    .filter( (col("Metric2") >= 0.6) & ( (col("is_current") == 1) ) ).selectExpr( 'IdComercializacion' )

estaciones_mal_portadas_2 = estaciones_mal_portadas_2_.alias("em") \
     .join( DeltaTable.forName(spark, 'analiticagdc.comercializacion.prophettunningparameter').toDF().alias("pp"), ( col("em.IdComercializacion") == col("pp.IdComercializacion") ) & (col("pp.is_current") == lit(True)), "left" ).filter( col("pp.Metric2") < 0.4  ).selectExpr( 'em.IdComercializacion' )

estaciones_mal_portadas = estaciones_mal_portadas_1.union(estaciones_mal_portadas_2)

estaciones_nuevas_hasta_ultima_tunning_prophet = DeltaTable.forName(spark, 'analiticagdc.comercializacion.dimestado').toDF() \
    .filter( (col("Estado") == 'NUEVA') & (col("is_current") == lit(True)) ).selectExpr( 'IdComercializacion' )

estaciones_media = estaciones_nuevas_hasta_ultima_tunning_prophet.union(estaciones_mal_portadas).drop_duplicates()


prophetpredictions = prophetpredictions_.filter( ~(col("IdComercializacion").isin( estaciones_media.rdd.flatMap(lambda x: x).collect() ) ) & ( col("Fecha") == fecha_ultimo_tuneo_prophet ) )
mediapredictions = mediapredictions_.filter( (col("IdComercializacion").isin( estaciones_media.rdd.flatMap(lambda x: x).collect() ) ) & ( col("Fecha") == fecha_ultimo_tuneo_prophet ) )

predictions_hoy = prophetpredictions.union(mediapredictions)
predictions_ = predictions_hoy.union(predictions_ayer)
predictions = predictions_.alias("p") \
    .join(estaciones.alias("e"), col("p.IdComercializacion") == col("e.Id"), "left") \
    .selectExpr( 'p.*', 
                 'descripcion as Estacion',
                 'tipo_usuario as TipoUsuario',
                 'id_electrocorrector as IdDispositivo')

results =  predictions.alias("p") \
    .join( real_values.alias("rv"), (col("p.IdComercializacion") == col("rv.IdComercializacion")) & (col("p.Fecha") == col("rv.Fecha")), 'left'  ) \
    .withColumn( 'Error', abs( col("p.Prediccion") - col("rv.Volumen") )/col("rv.Volumen") )  \
    .selectExpr( 'p.IdComercializacion',
                 'p.Estacion',
                 'p.IdDispositivo',
                 'p.TipoUsuario',
                 'p.Modelo',
                 'p.Fecha',
                 'rv.Volumen',
                 'p.Prediccion',
                 'Error'  ).filter( col('Fecha') >= fecha_penultimo_tuneo_prophet )


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
