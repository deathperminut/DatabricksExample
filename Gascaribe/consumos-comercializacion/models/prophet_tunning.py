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

def cuttof_maker(periods = 15, intra_period_lenght = 1):

    last_cutoff_date = datetime.now(timezone(timedelta(hours=-5), 'EST')) - timedelta(intra_period_lenght+1) 
    cuttof = [(last_cutoff_date - timedelta(intra_period_lenght*((periods-1)-i) )).strftime("%Y-%m-%d") for i in range(periods) ]

    return cuttof

# COMMAND ----------

def prophet_tunning(periods = 30, intra_period_lenght = 1):

    schema = StructType([
    StructField("estacion", IntegerType(), True),
    StructField("modelo", StringType(), True),
    StructField("rmse", FloatType(), True),
    StructField("metric_1", FloatType(), True),
    StructField("metric_2", FloatType(), True),
    StructField("metric", FloatType(), True),
    StructField("changepoint_prior_scale", FloatType(), True),
    ])

    @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
    def prophet_tunning_(df):  

        #periods = 60
        #intra_period_lenght = 1

        #    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        #    'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
        #, seasonality_prior_scale = params['seasonality_prior_scale'], holidays_prior_scale = params['holidays_prior_scale'] 
        df = df.copy()
        estacion_scores_rmse = []
        cutoffs = pd.to_datetime(cuttof_maker(periods = periods, intra_period_lenght = intra_period_lenght))


        try: 
            prophetdf = df.reset_index(drop=True).sort_values(by='Fecha')[['Fecha', 'VolumenCorregido']]
            prophetdf.columns = ['ds', 'y']
            prophetdf['ds'] = pd.to_datetime(prophetdf['ds'])

            holidays = df.loc[(df['Festivo'] == 1)].reset_index(drop=True).sort_values(by='Fecha')[['Fecha', 'VolumenCorregido']]
            holidays['holiday'] = 'Festivo'
            holidays = holidays[['holiday', 'Fecha']]
            holidays.columns = ['holiday', 'ds']
            holidays['ds'] = pd.to_datetime(holidays['ds'])

            param_grid = {  
                'changepoint_prior_scale': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
            }

            # Generate all combinations of parameters
            all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
            rmses = []  # Store the RMSEs for each params here
            metric = []  # Store the metric1's (percentage of days with less than 0.05) for each params here
            metric_1 = []  # Store the metric1's (percentage of days with less than 0.1) for each params here
            metric_2 = []  # Store the metric2's (percentage of days with less than 0.15) for each params here
            #print(df['Id'].iloc[0])
            # Use cross validation to evaluate all parameters
            for params in all_params:
                mp = Prophet(changepoint_prior_scale = params['changepoint_prior_scale'], holidays = holidays).fit(prophetdf)  # Fit model with given params
                df_cv = cross_validation(mp, cutoffs=cutoffs, horizon=f'{intra_period_lenght} days',parallel="processes")
                df_p2 = performance_metrics(df_cv, rolling_window=1)
                df_cv['percentual_error'] = np.abs( df_cv['yhat'] - df_cv['y']  )/df_cv['y']
                df_cv['Flag'] = np.where( df_cv['percentual_error'] < 0.1, 1, 0  ) 
                df_cv['Flag1'] = np.where( df_cv['percentual_error'] < 0.15, 1, 0  )
                df_cv['Flag2'] = np.where( df_cv['percentual_error'] < 0.05, 1, 0  ) 
                rmses.append(df_p2['rmse'].values[0])
                metric.append( np.sum(df_cv['Flag2'])/len(df_cv['Flag2']) )
                metric_1.append( np.sum(df_cv['Flag'])/len(df_cv['Flag']) )
                metric_2.append( np.sum(df_cv['Flag1'])/len(df_cv['Flag1']) )
            #print(df['Id'].iloc[0])
            # Find the best parameters
            tuning_results = pd.DataFrame(all_params)
            tuning_results['rmse'] = rmses
            tuning_results['metric_1'] = metric_1
            tuning_results['metric_2'] = metric_2
            tuning_results['metric'] = metric
            best_params = tuning_results.iloc[np.argmin(rmses)]
            estacion_scores_rmse.append({
                            'estacion': df['Id'].iloc[0],
                            'modelo': 'Prophet Forecasting',
                            'rmse': best_params['rmse'],
                            'metric_1': best_params['metric_1'],
                            'metric_2': best_params['metric_2'],
                            'metric': best_params['metric'],
                            'changepoint_prior_scale': best_params['changepoint_prior_scale']})
                #print('success')
        except Exception as e:
            estacion_scores_rmse.append({
                        'estacion': df['Id'].iloc[0],
                        'modelo': 'Prophet Forecasting',
                        'rmse': -1,
                        'metric_1': -1,
                        'metric_2': -1,
                        'metric': -1,
                        'changepoint_prior_scale': -1})

        return pd.DataFrame(estacion_scores_rmse)
    
    return prophet_tunning_




# COMMAND ----------

# MAGIC %md
# MAGIC ## Procedimiento

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE TABLE IF NOT EXISTS analiticagdc.comercializacion.prophettunningparameter (
# MAGIC   IdComercializacion       int,
# MAGIC   RMSE                     float,
# MAGIC   Metric                   float,
# MAGIC   Metric1                  float,
# MAGIC   Metric2                  float,
# MAGIC   Changepoint_prior_scale  float,
# MAGIC   FechaRegistro            date,
# MAGIC   is_current               boolean
# MAGIC )

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE TABLE IF NOT EXISTS analiticagdc.comercializacion.prophettunningparameter_06 (
# MAGIC   IdComercializacion       int,
# MAGIC   RMSE                     float,
# MAGIC   Metric                   float,
# MAGIC   Metric1                  float,
# MAGIC   Metric2                  float,
# MAGIC   Changepoint_prior_scale  float,
# MAGIC   FechaRegistro            date,
# MAGIC   is_current               boolean
# MAGIC )

# COMMAND ----------

insumo = DeltaTable.forName(spark, 'analiticagdc.comercializacion.insumo').toDF().filter( col("Estado") == lit('ACTIVA') ) \
    .withColumn('Id', col("IdComercializacion") ) \
    .selectExpr( "Fecha",
                 "IdComercializacion",
                 "Id",
                 "Estacion",
                 "Festivo",
                 "VolumenCorregido" )

# COMMAND ----------

# Partition the data
insumo.createOrReplaceTempView("insumo_pd")
sql = "select * from insumo_pd"
insumo_partition = (spark.sql(sql)\
   .repartition(spark.sparkContext.defaultParallelism, 
   ['IdComercializacion'])).cache()
insumo_partition.explain()

# COMMAND ----------

import logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

prophet_tunning_04 = prophet_tunning(periods = 30, intra_period_lenght = 1)
prophet_tunning_06 = prophet_tunning(periods = 60, intra_period_lenght = 1)

resultado_04 = insumo_partition.groupby(['IdComercializacion']).apply(prophet_tunning_04)
resultado_06 = insumo_partition.groupby(['IdComercializacion']).apply(prophet_tunning_06)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merge

# COMMAND ----------

prophet_tunning_parameters_04 = resultado_04 \
    .selectExpr( 'estacion as IdComercializacion',
                 'rmse as RMSE',
                 'metric as Metric',
                 'metric_1 as Metric1',
                 'metric_2 as Metric2',
                 'changepoint_prior_scale as Changepoint_prior_scale' )

# COMMAND ----------

prophet_tunning_parameters_06 = resultado_06 \
    .selectExpr( 'estacion as IdComercializacion',
                 'rmse as RMSE',
                 'metric as Metric',
                 'metric_1 as Metric1',
                 'metric_2 as Metric2',
                 'changepoint_prior_scale as Changepoint_prior_scale' )

# COMMAND ----------


prophet_tunning = DeltaTable.forName(spark, 'analiticagdc.comercializacion.prophettunningparameter').toDF().drop('is_current') \
    .withColumn('is_current', lit(False)) \

prophet_tunning.write.mode("overwrite").saveAsTable("analiticagdc.comercializacion.prophettunningparameter")

prophet_tunning_06 = DeltaTable.forName(spark, 'analiticagdc.comercializacion.prophettunningparameter_06').toDF().drop('is_current') \
    .withColumn('is_current', lit(False)) \

prophet_tunning_06.write.mode("overwrite").saveAsTable("analiticagdc.comercializacion.prophettunningparameter_06")


# COMMAND ----------

deltaTable_prophettunningparameter = DeltaTable.forName(spark, 'analiticagdc.comercializacion.prophettunningparameter')

insertar =  {
              "IdComercializacion"       : "df.IdComercializacion",
              "RMSE"                     : "df.RMSE",
              "Metric"                   : "df.Metric",
              "Metric1"                  : "df.Metric1",
              "Metric2"                  : "df.Metric2",
              "Changepoint_prior_scale"  : "df.Changepoint_prior_scale",
              "FechaRegistro"            : from_utc_timestamp(current_timestamp(), 'GMT-5'),
              "is_current"               : lit(True)
    }



deltaTable_prophettunningparameter.alias('t') \
  .merge( prophet_tunning_parameters_04.alias('df'), 'False') \
  .whenNotMatchedInsert(values=insertar) \
  .execute()

# COMMAND ----------

deltaTable_prophettunningparameter_06 = DeltaTable.forName(spark, 'analiticagdc.comercializacion.prophettunningparameter_06')

insertar =  {
              "IdComercializacion"       : "df.IdComercializacion",
              "RMSE"                     : "df.RMSE",
              "Metric"                   : "df.Metric",
              "Metric1"                  : "df.Metric1",
              "Metric2"                  : "df.Metric2",
              "Changepoint_prior_scale"  : "df.Changepoint_prior_scale",
              "FechaRegistro"            : from_utc_timestamp(current_timestamp(), 'GMT-5'),
              "is_current"               : lit(True)
    }



deltaTable_prophettunningparameter_06.alias('t') \
  .merge( prophet_tunning_parameters_06.alias('df'), 'False') \
  .whenNotMatchedInsert(values=insertar) \
  .execute()
