# Databricks notebook source
import os
import itertools
import pandas as pd
import numpy as np
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import *
from delta.tables import *
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import math
import random
import time
from datetime import date,datetime,timedelta,timezone

tomorrow = (datetime.now(timezone(timedelta(hours=-5), 'EST'))).strftime("%Y-%m-%d")
today = (datetime.now(timezone(timedelta(hours=-5), 'EST')) - timedelta(1))
two_days_prior = (datetime.now(timezone(timedelta(hours=-5), 'EST')) - timedelta(3)).strftime("%Y-%m-%d")
today_dt = today.strftime("%Y-%m-%d")
one_year_ago = (today - timedelta(days=365)).strftime("%Y-%m-%d")
two_years_ago = (today - timedelta(days=730)).strftime("%Y-%m-%d")


import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

print(two_days_prior)

# COMMAND ----------

# 78, 88, 468, 319           ((208, 234) Comienzan en 0 ) 
insumo = DeltaTable.forName(spark, "analiticagdc.comercializacion.insumo").toDF() \
.filter( col("Estado") == 'ACTIVA' ) \
.filter( col("Fecha") >= col("PrimeraFechaEfectiva") ) \
.orderBy(col("IdComercializacion"),col("Fecha"))

results = insumo.toPandas()
df = insumo.toPandas()

# COMMAND ----------

results[(results['IdComercializacion'] == 142) & (results['Volumen'] == results['VolumenCorregido']) ]

# COMMAND ----------

display(results[results['IdComercializacion'] == 142])

# COMMAND ----------

cutoffs = pd.to_datetime(['2024-01-22', '2024-01-30'])#'2024-01-23', '2024-02-06' 

df['Fecha'] = pd.to_datetime(df['Fecha'])


prophetdf = df[df['IdComercializacion'] == 351].reset_index(drop=True).sort_values(by='Fecha')[['Fecha', 'VolumenCorregido']]
#prophetdf.loc[(prophetdf['VolumenCorregido'] < 100) , 'VolumenCorregido'] = None
prophetdf.columns = ['ds', 'y']
prophetdf['ds'] = pd.to_datetime(prophetdf['ds'])
condition = np.where( (df['IdComercializacion'] == 351) & (df['Festivo'] == 1) )
holidays = df.loc[condition].reset_index(drop=True).sort_values(by='Fecha')[['Fecha', 'VolumenCorregido']]
holidays['holiday'] = 'Festivo'
holidays = holidays[['holiday', 'Fecha']]
holidays.columns = ['holiday', 'ds']
holidays['ds'] = pd.to_datetime(holidays['ds'])

# Modeling and forecasting
model = Prophet(changepoint_prior_scale = 0.5 , holidays = holidays)
model.fit(prophetdf)
future = model.make_future_dataframe(periods=0)
forecast = model.predict(future)
df_cv2 = cross_validation(model, cutoffs=cutoffs, horizon='2 days')
df_p = performance_metrics(df_cv2, rolling_window=1)
#future_dates = model.make_future_dataframe(periods=prophetTest.shape[0], freq='d')
#forecast = model.predict(future_dates)
           


# COMMAND ----------

forecast

# COMMAND ----------

from pandas.api.indexers import BaseIndexer
class CustomIndexer(BaseIndexer):
    def get_window_bounds(self, num_values, min_periods, center, closed, step):
        start = np.empty(num_values, dtype=np.int64)
        end = np.empty(num_values, dtype=np.int64)
        for i in range(num_values):
            #start[i] = i#
            #end[i] = i+1#(((i//self.window_size)+1)*self.window_size)-1
        
            start[i] = (i//(self.window_size))*(self.window_size)
            end[i] = np.min( [(start[i] + self.window_size), num_values-1] )

        return start, end

# COMMAND ----------

def sd_my(x):
    return np.sqrt( x.sum()/(len(x) - 1) )

# COMMAND ----------

forecast['y'] = prophetdf['y']
forecast['deviationsq'] = (forecast['y'] - forecast['trend'])*(forecast['y'] - forecast['trend'])
forecast['deviation'] = (forecast['y'] - forecast['trend'])
indexer = CustomIndexer(window_size=30)
indexer2 = CustomIndexer(window_size=15)
forecast['standar_deviation1'] = forecast['deviationsq'].rolling(indexer,  min_periods=1).apply(sd_my)
forecast['standar_deviation2'] = forecast['deviationsq'].rolling(indexer2,  min_periods=1).apply(sd_my)
forecast['y_upper_1'] = forecast['trend'] + forecast['standar_deviation1']
forecast['y_lower_1'] = forecast['trend'] - forecast['standar_deviation1']
forecast['y_upper_2'] = forecast['trend'] + forecast['standar_deviation2']
forecast['y_lower_2'] = forecast['trend'] - forecast['standar_deviation2']
forecast['outlier_flag'] = np.where( ( ( forecast['y'] > forecast['yhat_upper'] ) | ( forecast['y'] < forecast['yhat_lower'] ) ) & ( (1.5*forecast['standar_deviation1']) < (np.abs(forecast['deviation']))  ) & ( (1.5*forecast['standar_deviation2']) < (np.abs(forecast['deviation'])) ), True , False )
forecast['y_corregido'] = np.where( forecast['outlier_flag'] == True, None, forecast['y'] )

# COMMAND ----------

display(forecast[['ds', 'trend', 'standar_deviation1', 'standar_deviation2', 'y_upper_1', 'y_lower_1', 'y_upper_2',
'y_lower_2', 'y', 'deviation', 'outlier_flag', 'y_corregido']])

# COMMAND ----------

display(forecast[['ds', 'trend', 'Amplitud', 'OriginalMean']])

# COMMAND ----------

display(forecast[['ds', 'trend', 'yhat_upper', 'yhat_lower', 'y', 'y_upper_1', 'y_lower_1']])

# COMMAND ----------

fig = model.plot(forecast)

# COMMAND ----------

display(df_p) 

# COMMAND ----------

display(df_p)

# COMMAND ----------

display(df_p)

# COMMAND ----------

cutoffs = pd.to_datetime(['2024-01-23', '2024-01-30', '2024-02-06'])
Id = 88 # 14, 78, 88, 468, 319           ((208, 234) Comienzan en 0 ) 
prophetdf = df[df['IdComercializacion'] == Id].reset_index(drop=True).sort_values(by='Fecha')[['Fecha', 'VolumenCorregido']]
prophetdf.columns = ['ds', 'y']
prophetdf['ds'] = pd.to_datetime(prophetdf['ds'])

condition = np.where( (df['IdComercializacion'] == Id) & (df['Festivo'] == 1) )
holidays = df.loc[condition].reset_index(drop=True).sort_values(by='Fecha')[['Fecha', 'VolumenCorregido']]
holidays['holiday'] = 'Festivo'
holidays = holidays[['holiday', 'Fecha']]
holidays.columns = ['holiday', 'ds']
holidays['ds'] = pd.to_datetime(holidays['ds'])

# COMMAND ----------

# Resultados Cruzados sin tuneo  - para 300 estaciones
#    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
#    'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
#, seasonality_prior_scale = params['seasonality_prior_scale'], holidays_prior_scale = params['holidays_prior_scale'] 
estacion_scores_sintunning = []
cutoffs = pd.to_datetime(['2024-01-23', '2024-01-30', '2024-02-06'])
for idcomercializacion in df['IdComercializacion'].unique():

    try: 
        prophetdf = df[df['IdComercializacion'] == idcomercializacion].reset_index(drop=True).sort_values(by='Fecha')[['Fecha', 'VolumenCorregido']]
        prophetdf.columns = ['ds', 'y']
        prophetdf['ds'] = pd.to_datetime(prophetdf['ds'])

        condition = np.where( (df['IdComercializacion'] == idcomercializacion) & (df['Festivo'] == 1) )
        holidays = df.loc[condition].reset_index(drop=True).sort_values(by='Fecha')[['Fecha', 'VolumenCorregido']]
        holidays['holiday'] = 'Festivo'
        holidays = holidays[['holiday', 'Fecha']]
        holidays.columns = ['holiday', 'ds']
        holidays['ds'] = pd.to_datetime(holidays['ds'])


        # Use cross validation        
        mp = Prophet(holidays = holidays).fit(prophetdf)  # Fit model with given params
        df_cv = cross_validation(mp, cutoffs=cutoffs, horizon='1 days', parallel="processes")
        df_p2 = performance_metrics(df_cv, rolling_window=1)
        
        estacion_scores_sintunning.append({
                        'estacion': idcomercializacion,
                        'modelo': 'Prophet Forecasting',
                        'rmse': df_p2['rmse'].values[0],})
    except Exception as e:
            estacion_scores_sintunning.append({
                        'estacion': idcomercializacion,
                        'modelo': 'Prophet Forecasting',
                        'rmse': -1})


# COMMAND ----------

estacion_scores_sintunning_pd = pd.DataFrame(estacion_scores_sintunning)
volumenes_promedio = df[['IdComercializacion', 'VolumenCorregido']].groupby('IdComercializacion').mean('VolumenCorregido').reset_index()# 14, 78, 88, 468, 319,208, 234 

scores = estacion_scores_sintunning_pd.merge( volumenes_promedio, left_on= estacion_scores_sintunning_pd['estacion'], right_on= volumenes_promedio['IdComercializacion'] , how='left' )

scores['percentage'] = scores['rmse']/scores['VolumenCorregido']

# COMMAND ----------

display(scores)

# COMMAND ----------

# Tuneo Hiperparametros  - para 300 estaciones
#    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
#    'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
#, seasonality_prior_scale = params['seasonality_prior_scale'], holidays_prior_scale = params['holidays_prior_scale'] 
estacion_scores_rmse = []
cutoffs = pd.to_datetime(['2024-01-23', '2024-01-30', '2024-02-06'])
for idcomercializacion in df['IdComercializacion'].unique():

    try: 
        prophetdf = df[df['IdComercializacion'] == idcomercializacion].reset_index(drop=True).sort_values(by='Fecha')[['Fecha', 'VolumenCorregido']]
        prophetdf.columns = ['ds', 'y']
        prophetdf['ds'] = pd.to_datetime(prophetdf['ds'])

        condition = np.where( (df['IdComercializacion'] == idcomercializacion) & (df['Festivo'] == 1) )
        holidays = df.loc[condition].reset_index(drop=True).sort_values(by='Fecha')[['Fecha', 'VolumenCorregido']]
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

        # Use cross validation to evaluate all parameters
        for params in all_params:
            mp = Prophet(changepoint_prior_scale = params['changepoint_prior_scale'], holidays = holidays).fit(prophetdf)  # Fit model with given params
            df_cv = cross_validation(mp, cutoffs=cutoffs, horizon='1 days', parallel="processes")
            df_p2 = performance_metrics(df_cv, rolling_window=1)
            rmses.append(df_p2['rmse'].values[0])

        # Find the best parameters
        tuning_results = pd.DataFrame(all_params)
        tuning_results['rmse'] = rmses
        best_params = tuning_results.iloc[np.argmin(rmses)]
        estacion_scores_rmse.append({
                        'estacion': idcomercializacion,
                        'modelo': 'Prophet Forecasting',
                        'rmse': best_params['rmse'],
                        'changepoint_prior_scale': best_params['changepoint_prior_scale']})
    except Exception as e:
            estacion_scores_rmse.append({
                        'estacion': idcomercializacion,
                        'modelo': 'Prophet Forecasting',
                        'rmse': -1,
                        'changepoint_prior_scale': -1})


# COMMAND ----------

 display( pd.DataFrame(estacion_scores_rmse))

# COMMAND ----------

 display( pd.DataFrame(estacion_scores))

# COMMAND ----------

best_params = tuning_results.iloc[np.argmin(mapes)]
print(best_params)
#IdComercializacion 14 tuneando todo = 0.021767, solo_change_season = 0.022248, solo_change = 0.022817
#IdComercializacion 288 tuneando todo = 0.030165, solo_change_season = 0.030165, solo_change = 0.038807
#IdComercializacion 88 tuneando todo = 0.028614, solo_change_season = 0.028738, solo_change = 0.033978

# COMMAND ----------

display(df[['IdComercializacion', 'VolumenCorregido']].loc[np.where(df['IdComercializacion'].isin([184]))].groupby('IdComercializacion').mean('VolumenCorregido').reset_index())# 14, 78, 88, 468, 319,208, 234 

# COMMAND ----------

display(prophetTrain)

# COMMAND ----------

def predict_estaciones(df, two_years_ago, two_days_prior,today):
    # List to store station-wise scores
    estacion_scores = []

    # Filter data based on the given time frame
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df[df['Fecha'] >= two_years_ago]
    predicciones = 0
    
    # Iterate through unique stations (top 10)
    for idcomercializacion in df['IdComercializacion'].unique():
        print(f'PREDICTION FOR {idcomercializacion}')
        
        # Prepare data for Prophet
        prophetdf = df[df['IdComercializacion'] == idcomercializacion].reset_index(drop=True).sort_values(by='Fecha')[['Fecha', 'VolumenCorregido']]
        prophetdf.columns = ['ds', 'y']
        prophetdf['ds'] = pd.to_datetime(prophetdf['ds'])

        prophetTrain = prophetdf[prophetdf['ds'] < two_days_prior]
        prophetTest = prophetdf[prophetdf['ds'] >= two_days_prior]

        try:
            
            # Modeling and forecasting
            model = Prophet()
            model.fit(prophetTrain)

            future_dates = model.make_future_dataframe(periods=prophetTest.shape[0], freq='d')
            forecast = model.predict(future_dates)
           
            ##
            # Post-processing forecast results
            forecast['ds'] = pd.to_datetime(forecast['ds'])
            newForecast = forecast[forecast['ds'] >= two_days_prior][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            newForecast['yreal'] = prophetTest['y'].tolist()
            newForecast['error'] = np.abs(newForecast['yreal'] - newForecast['yhat'])
            newForecast['yhat'] = newForecast['yhat'].apply(lambda x: 0 if x < 0 else x)
            predicciones = newForecast.tail(1)['yhat'].values[0]
            
            # Calculating errors
            error = newForecast['error'].mean()
            errorDelDiaSiguiente = newForecast.iloc[0]['error']
            errorAbsolutoAlDiaSiguiente = np.abs((newForecast.iloc[0]['yreal'] - newForecast.iloc[0]['yhat']) / newForecast.iloc[0]['yhat']) * 100

            # Printing and storing scores
            print(f'MAE: {error:.2f} m3')
            print(f'Error for the next day: {errorDelDiaSiguiente:.2f} m3')
            print(f'Absolute error for the next day: {errorAbsolutoAlDiaSiguiente:.2f}%')

            estacion_scores.append({
                'estacion': idcomercializacion,
                'modelo': 'Prophet Forecasting',
                'mae': error,
                'fecha': tomorrow,
                'predicciones': predicciones,
                'error': errorDelDiaSiguiente,
                'error_absoluto': errorAbsolutoAlDiaSiguiente,
                'estado': 'Activa'
            })
        except Exception as e:
            estacion_scores.append({
                'estacion': idcomercializacion,
                'modelo': 'Prophet Forecasting',
                'mae': -1,
                'fecha': tomorrow,
                'predicciones': predicciones,
                'error': -1,
                'error_absoluto': -1,
                'estado': 'Activa'
            })

    return pd.DataFrame(estacion_scores)

# COMMAND ----------

display(prophet_results)

# COMMAND ----------

predicciones = prophet_results[['estacion','fecha','predicciones','error_absoluto','modelo','estado']]
predicciones['fecha'] = pd.to_datetime(predicciones['fecha'])
predicciones

# COMMAND ----------

schema = StructType([
    StructField("estacion", StringType(), True),
    StructField("fecha", DateType(), True),
    StructField("predicciones", FloatType(), True),
    StructField("error_absoluto", FloatType(), True),
    StructField("modelo", StringType(), True),
    StructField("estado", StringType(), True)
    ])
df = spark.createDataFrame(predicciones, schema = schema)

deltaTable = DeltaTable.forName(spark, 'analiticaefg.comercializacion.predicciones_prophet')

deltaTable.alias("t").merge(
    df.alias("s"),
    "t.estacion = s.estacion AND t.fecha = s.fecha"
).whenNotMatchedInsertAll().execute()
