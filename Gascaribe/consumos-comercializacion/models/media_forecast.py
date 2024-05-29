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

def media_loss_function(x, df, n=4, history_len=15):
     
    df = df.copy()
    
    try:
        predicciones = []
        valores_reales = []
        for i in range(history_len):

            prediccion = 0
            valor_real = df.reset_index(drop=True).sort_values(by='Fecha')['VolumenCorregido'].iloc[(i-history_len)]
            mediadf = df.reset_index(drop=True).sort_values(by='Fecha')[['Fecha', 'VolumenCorregido', 'DiaSemana', 'Festivo', 'Estado']].iloc[:(i-history_len)]
            mediadf['Fecha'] = pd.to_datetime(mediadf['Fecha'])
            
                
            day = mediadf.iloc[-1]['DiaSemana']
            festivo_flag = mediadf.iloc[-1]['Festivo']
            estado = mediadf.iloc[-1]['Estado']

            if (estado == 'NUEVA'):

                prediccion = mediadf[mediadf['DiaSemana'] == day].tail(n)['VolumenCorregido'].mean()

            elif festivo_flag == 1:

                prediccion = mediadf[mediadf['Festivo'] == 1].tail(n)['VolumenCorregido'].mean()
                
            elif (day in [0, 1, 2, 3, 4, 5, 6]) & (festivo_flag == 0):
                    
                prediccion = ( ((mediadf[mediadf['DiaSemana'] == day].tail(n)['VolumenCorregido'].mean() )*x[0]) + ( mediadf.tail(day+1)['VolumenCorregido'].mean()*x[1] ) )/2
            
            predicciones.append(prediccion)
            valores_reales.append(valor_real)
        
        predicciones_np = np.array(predicciones) 
        valores_reales_np = np.array(valores_reales)    

        rmse = np.sqrt( np.sum( (predicciones_np -  valores_reales_np)**2 )/len(predicciones_np) )
        
    except Exception as e:

        rmse = -1
            
    return rmse


# COMMAND ----------

def media_cross_validation(x, df, n=4, history_len=15):
     
    df = df.copy()
    
    try:
        predicciones = []
        valores_reales = []
        fecha_predicciones = []
        fechas = []
        for i in range(history_len):

            prediccion = 0
            valor_real = df.reset_index(drop=True).sort_values(by='Fecha')['VolumenCorregido'].iloc[(i-history_len)]
            fecha_prediccion = df.reset_index(drop=True).sort_values(by='Fecha')['Fecha'].iloc[(i-history_len)]
            fecha = df.reset_index(drop=True).sort_values(by='Fecha')['Fecha'].iloc[(i-history_len)-1]
            mediadf = df.reset_index(drop=True).sort_values(by='Fecha')[['Fecha', 'VolumenCorregido', 'DiaSemana', 'Festivo', 'Estado']].iloc[:(i-history_len)]
            mediadf['Fecha'] = pd.to_datetime(mediadf['Fecha'])
            
                
            day = mediadf.iloc[-1]['DiaSemana']
            festivo_flag = mediadf.iloc[-1]['Festivo']
            estado = mediadf.iloc[-1]['Estado']

            if (estado == 'NUEVA'):

                prediccion = mediadf[mediadf['DiaSemana'] == day].tail(n)['VolumenCorregido'].mean()

            elif festivo_flag == 1:

                prediccion = mediadf[mediadf['Festivo'] == 1].tail(n)['VolumenCorregido'].mean()
                
            elif (day in [0, 1, 2, 3, 4, 5, 6]) & (festivo_flag == 0):
                    
                prediccion = ( ((mediadf[mediadf['DiaSemana'] == day].tail(n)['VolumenCorregido'].mean() )*x[0]) + ( mediadf.tail(day+1)['VolumenCorregido'].mean()*x[1] ) )/2

            predicciones.append(prediccion)
            valores_reales.append(valor_real)
            fecha_predicciones.append(fecha_prediccion)
            fechas.append(fecha)
        
        predicciones_ = np.array(predicciones)
        predicciones_np = np.where( predicciones_<0, 0, predicciones_) 
        valores_reales_np = np.array(valores_reales)
        fecha_predicciones_np = np.array(fecha_predicciones)
        fechas_np = np.array(fechas)
        errores = np.abs( predicciones_np- valores_reales_np)/valores_reales_np

        rmse = np.sqrt( np.sum( (predicciones_np -  valores_reales_np)**2 )/len(predicciones_np) )


        
    except Exception as e:

        rmse = -1
            
    return pd.DataFrame({   'Fecha': fechas_np,
                            'FechaPrediccion': fecha_predicciones_np,
                            'ValorReal': valores_reales_np,
                            'Prediccion': predicciones_np,
                            'Error': errores})


# COMMAND ----------

def media_optimizer(n=4, history_len=15):

    schema = StructType([
        StructField("estacion", IntegerType(), True),
        StructField("modelo", StringType(), True),
        StructField("rmse", FloatType(), True),
        StructField("w1", FloatType(), True),
        StructField("w2", FloatType(), True),
        StructField("metric", FloatType(), True),
        StructField("metric_1", FloatType(), True),
        StructField("metric_2", FloatType(), True),
        ])

    @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
    def media_optimizer_(df):

        df = df.copy()
        media_rmse_result = []

        try:
        
            

            mediadf = df.reset_index(drop=True)
            resultado = minimize( media_loss_function, x0 = [0.5, 0.5], method='SLSQP', tol=1e-6, args=(mediadf, n, history_len) )
    
            w1 = resultado['x'][0]
            w2 = resultado['x'][1]
            df_cv = media_cross_validation(x=[w1, w2], df=mediadf, n=n, history_len=history_len)
            df_cv['Flag'] = np.where( df_cv['Error'] < 0.1, 1, 0  ) 
            df_cv['Flag1'] = np.where( df_cv['Error'] < 0.15, 1, 0  )
            df_cv['Flag2'] = np.where( df_cv['Error'] < 0.05, 1, 0  ) 
            metric =  np.sum(df_cv['Flag2'])/len(df_cv['Flag2'])    #metric's (percentage of days with less than 0.05) 
            metric_1 =  np.sum(df_cv['Flag'])/len(df_cv['Flag']) #metric1's (percentage of days with less than 0.1) 
            metric_2 =  np.sum(df_cv['Flag1'])/len(df_cv['Flag1']) #metric2's (percentage of days with less than 0.15) 




            media_rmse_result.append({
                            'estacion': df['Id'].iloc[0],
                            'modelo': 'Media Forecasting',
                            'rmse': resultado['fun'],
                            'w1': w1,
                            'w2': w2,
                            'metric': metric,
                            'metric_1': metric_1,
                            'metric_2': metric_2})

        except Exception as e:

            media_rmse_result.append({
                                'estacion': df['Id'].iloc[0],
                                'modelo': 'Media Forecasting',
                                'rmse': -1,
                                'w1': -1,
                                'w2': -1,
                                'metric': -1,
                                'metric_1': -1,
                                'metric_2': -1})



        return pd.DataFrame(media_rmse_result)
    
    return media_optimizer_


# COMMAND ----------

def media_forecast(n=4):

    schema = StructType([ StructField("IdComercializacion", IntegerType(), True),
                          StructField("Modelo", StringType(), True),
                          StructField("Fecha", DateType(), True),
                          StructField("Prediccion", FloatType(), True), ])

    @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
    def media_forecast_(df):

        df = df.copy()
        media_forecast = []

        try:


            prediccion = 0
            mediadf = df.reset_index(drop=True).sort_values(by='Fecha')[['Fecha', 'VolumenCorregido', 'DiaSemana', 'Festivo', 'Estado']]
            mediadf['Fecha'] = pd.to_datetime(mediadf['Fecha'])

            day = mediadf.iloc[-1]['DiaSemana']
            festivo_flag = mediadf.iloc[-1]['Festivo']
            estado = mediadf.iloc[-1]['Estado']

            if (estado == 'NUEVA'):

                prediccion = mediadf[mediadf['DiaSemana'] == day].tail(n)['VolumenCorregido'].mean()

            elif festivo_flag == 1:

                prediccion = mediadf[mediadf['Festivo'] == 1].tail(n)['VolumenCorregido'].mean()

            elif (day in [0, 1, 2, 3, 4, 5, 6]) & (festivo_flag == 0):

                w1 = df['w1'].iloc[0] #media_parameters[ (media_parameters['estacion'] == idcomercializacion)]['w1'].values[0]
                w2 = df['w2'].iloc[0] #media_parameters[ (media_parameters['estacion'] == idcomercializacion)]['w2'].values[0]
                prediccion = ( ( (mediadf[mediadf['DiaSemana'] == day].tail(n)['VolumenCorregido'].mean() )*w1 ) + ( mediadf.tail(day+1)['VolumenCorregido'].mean()*w2 ) )/2

            if prediccion < 0:
                prediccion = 0

            media_forecast.append({
                                'IdComercializacion': df['Id'].iloc[0],
                                'Modelo': 'Media',
                                'Fecha': (mediadf.iloc[-1]['Fecha'] + timedelta(1)),
                                'Prediccion': prediccion })
        except Exception as e:

            media_forecast.append({
                                'IdComercializacion': df['Id'].iloc[0],
                                'Modelo': 'Media',
                                'Fecha':  (mediadf.iloc[-1]['Fecha'] + timedelta(1)),
                                'Prediccion': -1 })

        return pd.DataFrame(media_forecast)
    
    return media_forecast_




# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS analiticagdc.comercializacion.mediapredictions (
# MAGIC   IdComercializacion       int,
# MAGIC   Modelo                   string,
# MAGIC   Fecha                    date,
# MAGIC   Prediccion               float
# MAGIC )

# COMMAND ----------

insumo = DeltaTable.forName(spark, 'analiticagdc.comercializacion.insumo').toDF() \
    .withColumn('Id', col("IdComercializacion") ) 
    
estaciones_mal_portadas = DeltaTable.forName(spark, 'analiticagdc.comercializacion.prophettunningparameter').toDF() \
    .filter( (col("Metric2") < 0.4) & (col("is_current") == 1) ).selectExpr( 'IdComercializacion' )


fechas_tuneo = DeltaTable.forName(spark, 'analiticagdc.comercializacion.prophettunningparameter').toDF() \
.selectExpr( 'FechaRegistro' ).dropDuplicates().orderBy(desc(col('FechaRegistro')))
fecha_ultimo_tuneo_prophet = fechas_tuneo.collect()[0][0] 

try:
    fecha_penultimo_tuneo_prophet = fechas_tuneo.collect()[1][0]
except Exception as e:
    fecha_penultimo_tuneo_prophet = to_date(lit("1900-01-01")) 

estaciones_nuevas_hasta_ultima_tunning_prophet = DeltaTable.forName(spark, 'analiticagdc.comercializacion.dimestado').toDF() \
    .filter( ( (col("Estado") == 'NUEVA') & (col("is_current") == lit(True)) ) 
             | ( (col("Estado") == 'NUEVA') & (col("FechaRegistro") <= fecha_ultimo_tuneo_prophet)
                & (col("FechaRegistro") >= fecha_penultimo_tuneo_prophet) ) ).selectExpr( 'IdComercializacion' )

insumo_nuevas = insumo.filter( (col("IdComercializacion").isin( estaciones_nuevas_hasta_ultima_tunning_prophet.rdd.flatMap(lambda x: x).collect() ) ) ) \
                      .selectExpr("Fecha",
                                  "IdComercializacion",
                                  "DiaSemana",
                                  "Id",
                                  "Estacion",
                                  "Festivo",
                                  "VolumenCorregido" ).withColumn('Estado', lit("NUEVA"))
                     
insumo_mal_portadas = insumo.filter( (col("IdComercializacion").isin( estaciones_mal_portadas.rdd.flatMap(lambda x: x).collect() ) ) ) \
                            .selectExpr( "Fecha",
                                        "IdComercializacion",
                                        "DiaSemana",
                                        "Id",
                                        "Estacion",
                                        "Festivo",
                                        "VolumenCorregido", "Estado" )

insumo_media = insumo_nuevas.union(insumo_mal_portadas)

# COMMAND ----------

# Partition the data
insumo_mal_portadas.createOrReplaceTempView("insumo_mp")
sql = "select * from insumo_mp"
insumo_mp_partition = (spark.sql(sql)\
   .repartition(spark.sparkContext.defaultParallelism, 
   ['IdComercializacion'])).cache()


# COMMAND ----------

media_optimizer_4_15 = media_optimizer(n=4, history_len=15)
media_parametros = insumo_mp_partition.groupby(['IdComercializacion']).apply(media_optimizer_4_15)  

# COMMAND ----------

insumo_media_parametros = insumo_media.alias("im") \
    .join( media_parametros.alias("mp"), col("im.IdComercializacion") == col("mp.estacion"), "left" ) \
    .selectExpr( "im.*", "mp.w1", "mp.w2" )    


insumo_media_parametros.createOrReplaceTempView("insumo_m")
sql = "select * from insumo_m"
insumo_media_parametros_partition = (spark.sql(sql)\
   .repartition(spark.sparkContext.defaultParallelism, 
   ['IdComercializacion'])).cache()

# COMMAND ----------

media_forecast_4 = media_forecast(n=4)
forecast_sdf = insumo_media_parametros_partition.groupby(['IdComercializacion']).apply(media_forecast_4) 

# COMMAND ----------

deltaTable_mediapredictions = DeltaTable.forName(spark, 'analiticagdc.comercializacion.mediapredictions')

mapping =  {
              "IdComercializacion"  : "df.IdComercializacion",
              "Modelo"              : "df.Modelo",
              "Fecha"               : "df.Fecha",
              "Prediccion"          : "df.Prediccion"
    }

deltaTable_mediapredictions.alias('t') \
  .merge( forecast_sdf.alias('df'), 't.IdComercializacion = df.IdComercializacion AND t.Fecha = df.Fecha') \
  .whenMatchedUpdate(set=mapping) \
  .whenNotMatchedInsert(values=mapping) \
  .execute()
