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
from pandas.api.indexers import BaseIndexer

import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Funciones

# COMMAND ----------

def prophet_forecast(df, parametros):
    
    df = df.copy()
    parametros = parametros.copy()
    prophet_forecast = []

    for idcomercializacion in df['IdComercializacion'].unique():

        try:
            
            prophetdf = df[df['IdComercializacion'] == idcomercializacion].reset_index(drop=True).sort_values(by='Fecha')[['Fecha', 'VolumenCorregido']]
            prophetdf.columns = ['ds', 'y']
            prophetdf['ds'] = pd.to_datetime(prophetdf['ds'])

            holidays = df.loc[(df['IdComercializacion'] == idcomercializacion) & (df['Festivo'] == 1)].reset_index(drop=True).sort_values(by='Fecha')[['Fecha', 'VolumenCorregido']]
            holidays['holiday'] = 'Festivo'
            holidays = holidays[['holiday', 'Fecha']]
            holidays.columns = ['holiday', 'ds']
            holidays['ds'] = pd.to_datetime(holidays['ds'])
            
            parameters = parametros[parametros['IdComercializacion'] == idcomercializacion].reset_index(drop=True)
            changepoint_parameter = parameters['Changepoint_prior_scale'].values[0]


            mp = Prophet(changepoint_prior_scale = changepoint_parameter, holidays = holidays).fit(prophetdf)
            future = mp.make_future_dataframe(periods=1)
            forecast = mp.predict(future)
            forecast['prediccion'] = np.where( forecast['yhat'] < 0, 0, forecast['yhat'] )

            prophet_forecast.append({
                                'IdComercializacion': idcomercializacion,
                                'Modelo': 'Prophet',
                                'Fecha': forecast['ds'].iloc[-1],
                                'Prediccion': forecast['prediccion'].iloc[-1] })
        except Exception as e:

            prophet_forecast.append({
                                'IdComercializacion': idcomercializacion,
                                'Modelo': 'Prophet',
                                'Fecha': -1,
                                'Prediccion': -1 })
            
    return pd.DataFrame(prophet_forecast)




# COMMAND ----------

# MAGIC %md
# MAGIC ## Procedimiento

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS analiticagdc.comercializacion.prophetpredictions (
# MAGIC   IdComercializacion       int,
# MAGIC   Modelo                   string,
# MAGIC   Fecha                    date,
# MAGIC   Prediccion               float
# MAGIC )

# COMMAND ----------

insumo = DeltaTable.forName(spark, 'analiticagdc.comercializacion.insumo').toDF() 
parametros = DeltaTable.forName(spark, 'analiticagdc.comercializacion.prophettunningparameter').toDF() \
    .filter( col("is_current") == lit(True))
insumo_pd = insumo.toPandas()
parametros_pd = parametros.toPandas()
insumo_pd_activa = insumo_pd[insumo_pd['Estado'] == 'ACTIVA'].reset_index(drop=True)

# COMMAND ----------

forecast = prophet_forecast(df=insumo_pd_activa,parametros=parametros_pd)

# COMMAND ----------

schema = StructType([
    StructField("IdComercializacion", IntegerType(), True),
    StructField("Modelo", StringType(), True),
    StructField("Fecha", DateType(), True),
    StructField("Prediccion", FloatType(), True),
    ])

forecast_sdf = spark.createDataFrame(forecast, schema = schema)

# COMMAND ----------

deltaTable_prophetpredictions = DeltaTable.forName(spark, 'analiticagdc.comercializacion.prophetpredictions')

mapping =  {
              "IdComercializacion"  : "df.IdComercializacion",
              "Modelo"              : "df.Modelo",
              "Fecha"               : "df.Fecha",
              "Prediccion"          : "df.Prediccion"
    }

deltaTable_prophetpredictions.alias('t') \
  .merge( forecast_sdf.alias('df'), 't.IdComercializacion = df.IdComercializacion AND t.Fecha = df.Fecha') \
  .whenMatchedUpdate(set=mapping) \
  .whenNotMatchedInsert(values=mapping) \
  .execute()
