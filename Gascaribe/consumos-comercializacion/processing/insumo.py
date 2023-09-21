# Databricks notebook source
import os
import pandas as pd
import numpy as np
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, DateType
from pyspark.sql import SparkSession
from pyspark.sql.functions import max as pySparkMax
from pyspark.sql.functions import col
from delta.tables import DeltaTable
import random
import scipy.stats as stats
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

# COMMAND ----------

# MAGIC %md
# MAGIC #### **Lista de festivos**

# COMMAND ----------

years = [2018,2019,2021,2022,2023,2024]
festivos = []

for year in years:
    colombia_holidays = holidays.Colombia(years=year)
    festivos += [x.strftime("%Y-%m-%d") for x in colombia_holidays.keys()]

# COMMAND ----------

# MAGIC %md
# MAGIC #### **Lista de ids de Comercializacion**

# COMMAND ----------

## Llenar

# COMMAND ----------

# MAGIC %md
# MAGIC #### **Ingesta de datos**

# COMMAND ----------

spark = SparkSession.builder \
    .appName("DeltaLake") \
    .config("spark.some.config.option", "config-value") \
    .getOrCreate()

ingesta = DeltaTable.forName(spark, "analiticagdc.comercializacion.ingesta").toDF()
estado = DeltaTable.forName(spark, "analiticagdc.comercializacion.estado").toDF()

t1 = estado.groupBy("estacion").agg(pySparkMax("fecharegistro").alias("latest_fecharegistro"))

estado_join = t1.alias('t1').join(estado.alias('e'), (t1.estacion == estado.estacion) & (t1.latest_fecharegistro == estado.fecharegistro),'inner').select('t1.estacion','e.estado','t1.latest_fecharegistro')



df_ingesta = ingesta.toPandas()
df_estado = estado_join.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC #### **Funciones de procesamiento**

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
        #_['id'] = _['id'].fillna(_['id'].mode()[0])
        _['tipo'] = _['tipo'].fillna(_['tipo'].mode()[0])
        _['volumenm3'] = _['volumenm3'].fillna(0)

        newdf = pd.concat([_,newdf],axis=0)
    
    return newdf

# COMMAND ----------

def process_inputs(df,today=today_dt):
    df = df.copy()

    df = df[df['fecha'] < today_cot]
    # Reemplazar los IdDispositivo por los ids
    #df['id'] = df['iddispositivo'].map(map_new_ids)

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
    #df['id'] = df['id'].astype('int')

    df = df.drop(['iddispositivo','tipo'],axis=1)

    return df

# COMMAND ----------

X = process_inputs(df_ingesta)

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
        vol_array = df[df['estacion'] == estacion].reset_index(drop=True).tail(nDaysNew)['volumenm3'].values

        len_vol_array = (len(vol_array))

        count = 0
        for vol in vol_array:
            if vol == 0:
                count += 1
        
        if (count >= minDays) or (len_vol_array < 15):
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

status_active_df = active_station_criterion(X,df_estado)

# COMMAND ----------

def new_station_criterion(df,estados,nDaysNew=30,percentage=0.9):
    df = df.copy()
    df_inactivas = estados[estados['estado'] == 'Inactiva']

    minDays = int(nDaysNew*percentage) # 30*0.9 = 27
    maxNumCount = nDaysNew - minDays # 30 - 27 = 3 (numero maximo de 0s)
    estacionesStatus = {}
    estaciones = df_inactivas['estacion'].unique()

    for estacion in estaciones:
        vol_array = df[df['estacion'] == estacion].reset_index(drop=True).tail(nDaysNew)['volumenm3'].values
        
        len_vol_array = len(vol_array)

        count = 0
        for vol in vol_array:
            if vol == 0:
                count += 1
        #print(count)
        if (count >= maxNumCount) or (len_vol_array < 30):
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

status_inactive_df = new_station_criterion(X,df_estado)

# COMMAND ----------

status_df = pd.concat([status_active_df,status_inactive_df],axis=0)

# COMMAND ----------

def hampel_filter(df,window_size,n=4,num_devs=3.0):
    
    cols = list(df.columns) + ['consumocorregido']
    newdf = pd.DataFrame(columns=cols)

    #tomorrow_day = tomorrow.weekday()

    estaciones = df['estacion'].unique()
    for estacion in estaciones:
        dataDF = df[df['estacion'] == estacion].fillna(0).reset_index(drop=True)
        data = np.array(dataDF['volumenm3'])

        filtered_data = data.copy()
        

        for i in range(len(data)):
            lower_bound = max(0, i - window_size)
            upper_bound = min(len(data), i + window_size)

            window = data[lower_bound:upper_bound]
            median = np.median(window)
            mad = np.median(np.abs(window - median))
            threshold = num_devs * 1.4826 * mad  # Factor of 1.4826 makes the MAD scale estimate consistent with std deviation

            if np.abs(data[i] - median) > threshold:
                day = dataDF.iloc[i]['diadesemana']
                listOfValues = list(dataDF[dataDF['diadesemana'] == day].tail(n)['volumenm3'])
                meanValues = sum(listOfValues)/n
                #print(meanValues)
                filtered_data[i] = meanValues
                #print(f'Indice reemplazado: {i}')

        dataDF['consumocorregido'] = filtered_data

        newdf = pd.concat([dataDF,newdf],axis=0)

    return newdf
    

# COMMAND ----------

X = hampel_filter(X,window_size=100)

# COMMAND ----------

status_df[status_df['estacion'] == 'CARTON COLOMBIA(MOLINO 5)']

# COMMAND ----------

# MAGIC %md
# MAGIC #### **Escritura de datos en el DeltaLake**

# COMMAND ----------

schema = StructType([
    StructField("estacion", StringType(), True),
    StructField("fecha", DateType(), True),
    StructField("volumenm3", FloatType(), True),
    StructField("festivos", IntegerType(), True),
    StructField("diadesemana", IntegerType(), True),
    StructField("consumocorregido", FloatType(), True)
    ])
df = spark.createDataFrame(X, schema = schema)

deltaTable = DeltaTable.forName(spark, 'analiticagdc.comercializacion.insumo')

deltaTable.alias("t").merge(
    df.alias("s"),
    "t.estacion = s.estacion AND t.fecha = s.fecha AND t.volumenm3 = s.volumenm3"
).whenNotMatchedInsertAll().execute()

# COMMAND ----------

schema = StructType([
    StructField("estacion", StringType(), True),
    StructField("estado", StringType(), True),
    StructField("fecharegistro", DateType(), True)
    ])
df = spark.createDataFrame(status_df, schema = schema)

deltaTable = DeltaTable.forName(spark, 'analiticagdc.comercializacion.estado')

deltaTable.alias("t").merge(
    df.alias("s"),
    "t.estacion = s.estacion AND t.estado = s.estado AND t.fecharegistro = s.fecharegistro"
).whenNotMatchedInsertAll().execute()

# COMMAND ----------


