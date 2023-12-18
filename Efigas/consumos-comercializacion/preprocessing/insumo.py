# Databricks notebook source
import os
import pandas as pd
import numpy as np
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, DateType
from pyspark.sql import SparkSession
from delta.tables import DeltaTable
import random
import scipy.stats as stats
from prophet import Prophet
from datetime import date,datetime,timedelta
import pytz
import holidays
cot_timezone = pytz.timezone('America/Bogota') # Saca el dia, teniendo en cuenta el timezone de Colombia
today = datetime.now()
today_cot = today.astimezone(cot_timezone).date()
tomorrow = today + timedelta(days=1)
today_dt = today_cot.strftime("%d-%m-%Y")

import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

import logging

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").disabled=True

estacionesNoUtilizables = [45, 107]

# COMMAND ----------

years = [2018,2019,2021,2022,2023,2024]
festivos = []

for year in years:
    colombia_holidays = holidays.Colombia(years=year)
    festivos += [x.strftime("%Y-%m-%d") for x in colombia_holidays.keys()]

# COMMAND ----------

storage_account_name = dbutils.secrets.get(scope='efigas', key='bs-name')
blob_container = dbutils.secrets.get(scope='efigas', key='bs-container')
blob_storage = storage_account_name + ".blob.core.windows.net"
config_key = "fs.azure.account.key."+storage_account_name+".blob.core.windows.net"
blob_access_key = dbutils.secrets.get(scope='efigas', key='bs-access-key')
spark.conf.set(config_key, blob_access_key)

# COMMAND ----------

ingesta = DeltaTable.forName(spark, f"analiticaefg.comercializacion.ingesta").toDF()
estados = spark.sql("SELECT * FROM analiticaefg.comercializacion.estado WHERE fecharegistro = (SELECT MAX(fecharegistro) FROM analiticaefg.comercializacion.estado)")

ingesta = ingesta.toPandas()
estados = estados.toPandas()

# COMMAND ----------

ingesta

# COMMAND ----------

def dummyDates(df):
    # Creacion de la tabla de fechas completas por estacion
    crossTable = pd.DataFrame(columns=['estacion','fecha'])
    for estacion in df['estacion'].unique():
        tempTable = pd.DataFrame(columns=['estacion','fecha'])
        _ = df[df['estacion'] == estacion].sort_values(by='fecha',ascending=True)
        startDate = _.head(1)["fecha"].values[0]
        endDate = _.tail(1)["fecha"].values[0]
        
        dateArray = []

        while startDate <= endDate:
            dateArray.append(startDate.strftime('%Y-%m-%d'))
            startDate += timedelta(days=1)

        tempTable['fecha'] = dateArray
        tempTable['estacion'] = [estacion]*len(dateArray)
        

        crossTable = pd.concat([tempTable,crossTable],axis=0)
    else:
        pass

    return crossTable

# COMMAND ----------

def fillNaN(df):
    newdf = pd.DataFrame(columns=df.columns)
    for estacion in df['estacion'].unique():
        _ = df[df['estacion'] == estacion].sort_values(by='fecha',ascending=True)

        _['iddispositivo'] = _['iddispositivo'].fillna(_['iddispositivo'].mode()[0])
        _['idcomercializacion'] = _['idcomercializacion'].fillna(_['idcomercializacion'].mode()[0])
        _['tipo'] = _['tipo'].fillna(_['tipo'].mode()[0])
        _['volumenm3'] = _['volumenm3'].fillna(0)

        newdf = pd.concat([_,newdf],axis=0)
    
    return newdf

# COMMAND ----------

def process_inputs(df,today=today_dt):
    df = df.copy()

    df = df[df['fecha'] < today_cot]
    # Reemplazar los IdDispositivo por los ids

    # Creacion de tabla de fechas completas por estacion
    crossTable = dummyDates(df)

    # Reemplazar las fechas::str por fechas::datetime
    df['fecha'] = pd.to_datetime(df['fecha'])
    crossTable['fecha'] = pd.to_datetime(crossTable['fecha'])

    # JOIN las dos tablas anteriores
    df = crossTable.merge(df,how='left',on=['estacion','fecha'])

    # Crear el flag de festivos
    festivosBin = []
    for fecha in df['fecha']:
        if str(fecha) in festivos:
            festivosBin.append(1)
        else:
            festivosBin.append(0)
            
    df['festivos'] = festivosBin

    # Crear la columna de dias de semana
    df['diadesemana'] = df['fecha'].apply(lambda x: x.dayofweek)

    # Crear el 8vo dia de la semana
    for i,festivo in enumerate(df['festivos']):
        if festivo == 1:
            df['diadesemana'][i] = 8
        else:
            pass

    # Reemplazar los valores NaN:
    # IdDispositivo: Moda
    # id: Moda
    # Tipo: Moda
    # VolumenM3: 0
    df = fillNaN(df)

    df['volumenm3'] = df['volumenm3'].astype('float')
    df['festivos'] = df['festivos'].astype('int')
    df['idcomercializacion'] = df['idcomercializacion'].astype('int')

    df = df.drop(['iddispositivo','tipo'],axis=1)

    return df

# COMMAND ----------

X = process_inputs(ingesta)

# COMMAND ----------

# Esta funcion deberia detectar si en los ultimos nDaysNew, la estacion ha tenido mas de percentage*nDaysNew dias 
# con consumos en 0. Si tiene percentage*nDaysNew o mas dias en 0, se le deberia agregar un flag de inactividad.
def active_station_criterion(df,estados,nDaysNew=15,percentage=0.9):
    df = df.copy()
    df_activas = estados[estados['estado'] == 'Activa']

    minDays = int(nDaysNew*percentage)
    estacionesStatus = {}
    estaciones = df_activas['estacion'].unique()

    for estacion in estaciones:
        volArray = df[df['estacion'] == estacion].reset_index(drop=True).tail(nDaysNew)['volumenm3'].values

        count = 0
        for vol in volArray:
            if vol == 0:
                count += 1
        
        if count >= minDays:
            estacionesStatus[estacion] = 'Inactiva'
        else:
            estacionesStatus[estacion] = 'Activa'

    
    estacion = []
    estado = []
    for key,value in estacionesStatus.items():
        estacion.append(key)
        estado.append(value)
    
    status_dict = {}
    status_dict['estacion'] = estacion
    status_dict['estado'] = estado
    status_dict['fecharegistro'] = today_cot

    status_df = pd.DataFrame(status_dict)

    #newdf = df.merge(status_df[['estacion','estado']],how='left',on='estacion')

        

    return status_df

# COMMAND ----------

status_active_df = active_station_criterion(X,estados)

# COMMAND ----------

def new_station_criterion(df,estados,nDaysNew=30,percentage=0.9):
    df = df.copy()
    df_inactivas = estados[estados['estado'] == 'Inactiva']

    minDays = int(nDaysNew*percentage) # 30*0.9 = 27
    maxNumCount = nDaysNew - minDays # 30 - 27 = 3 (numero maximo de 0s)
    estacionesStatus = {}
    estaciones = df_inactivas['estacion'].unique()

    for estacion in estaciones:
        volArray = df[df['estacion'] == estacion].reset_index(drop=True).tail(nDaysNew)['volumenm3'].values
        #print(volArray)
        count = 0
        for vol in volArray:
            if vol == 0:
                count += 1
        #print(count)
        if count >= maxNumCount:
            estacionesStatus[estacion] = 'Nueva'
        else:
            estacionesStatus[estacion] = 'Activa'

    estacion = []
    estado = []
    for key,value in estacionesStatus.items():
        estacion.append(key)
        estado.append(value)
    
    status_dict = {}
    status_dict['estacion'] = estacion
    status_dict['estado'] = estado
    status_dict['fecharegistro'] = today_cot

    status_df = pd.DataFrame(status_dict)

    return status_df
        

# COMMAND ----------

status_inactive_df = new_station_criterion(X,estados)

# COMMAND ----------

status_df = pd.concat([status_active_df,status_inactive_df],axis=0)

# COMMAND ----------

def prophet_filter(df,periods,sd=3,n=4):
    df = df.copy()
    df['ds'] = df['fecha']
    df['y'] = df['volumenm3']
    new_df = pd.DataFrame()
    
    for est in df['estacion'].unique():
        print(est)
        _ = df[(df['estacion'] == est)]
        model = Prophet()
        model.fit(_[['ds','y']])

        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)[['ds','yhat']]
        residuals_df = _.merge(forecast,how='left',on='ds')
        residuals_df['residuals'] = residuals_df['y'] - residuals_df['yhat']
        residuals = residuals_df['y'] - residuals_df['yhat']

        std_dev = residuals.std()
        mean_residual = residuals.mean()
        threshold = 3 * std_dev

        residuals_df['consumocorregido'] = 0

        for i in range(len(residuals_df)):
            if residuals_df['residuals'][i] > threshold:
                day = residuals_df.iloc[i]['diadesemana']
                listOfValues = list(residuals_df[residuals_df['diadesemana'] == day].tail(n)['y'])
                meanValues = sum(listOfValues)/n
                residuals_df['consumocorregido'][i] = meanValues

            else:
                residuals_df['consumocorregido'][i] = residuals_df['y'][i]
            
        new_df = pd.concat([residuals_df,new_df],axis=0)
        #new_df['window_size'] = periods

    print(forecast.columns)
    
    return new_df

# COMMAND ----------

new_X = prophet_filter(X,periods=10)

# COMMAND ----------

new_X.columns

# COMMAND ----------

results = new_X[['estacion','fecha','volumenm3','idcomercializacion','festivos','diadesemana','consumocorregido','residuals']]

# COMMAND ----------

schema = StructType([
    StructField("estacion", StringType(), True),
    StructField("fecha", DateType(), True),
    StructField("volumenm3", FloatType(), True),
    StructField("idcomercializacion", IntegerType(), True),
    StructField("festivos", IntegerType(), True),
    StructField("diadesemana", IntegerType(), True),
    StructField("volumen_corregido", FloatType(), True),
    StructField("residuals", FloatType(), True),
    ])
df = spark.createDataFrame(results, schema = schema)

deltaTable = DeltaTable.forName(spark, 'analiticaefg.comercializacion.insumo')

deltaTable.alias("t").merge(
    df.alias("s"),
    "t.estacion = s.estacion AND t.fecha = s.fecha AND t.volumenm3 = s.volumenm3"
).whenNotMatchedInsertAll().execute()

# COMMAND ----------


