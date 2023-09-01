# Databricks notebook source
import os
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, DateType
import random
import scipy.stats as stats
from statsmodels.tsa.stattools import grangercausalitytests
from datetime import date,datetime
import holidays
today = datetime.now()
today_dt = today.strftime("%d-%m-%Y")

import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

# COMMAND ----------

years = [2018,2019,2021,2022,2023,2024]
festivos = []

for year in years:
    colombia_holidays = holidays.Colombia(years=year)
    festivos += [x.strftime("%Y-%m-%d") for x in colombia_holidays.keys()]

# COMMAND ----------

dwDatabase = dbutils.secrets.get(scope='gascaribe', key='dwh-name')
dwServer = dbutils.secrets.get(scope='gascaribe', key='dwh-host')
dwUser = dbutils.secrets.get(scope='gascaribe', key='dwh-user')
dwPass = dbutils.secrets.get(scope='gascaribe', key='dwh-pass')
dwJdbcPort = dbutils.secrets.get(scope='gascaribe', key='dwh-port')
dwJdbcExtraOptions = ""
sqlDwUrl = "jdbc:sqlserver://" + dwServer + ".database.windows.net:" + dwJdbcPort + ";database=" + dwDatabase + ";user=" + dwUser + ";password=" + dwPass + ";" + dwJdbcExtraOptions
storage_account_name = dbutils.secrets.get(scope='gascaribe', key='bs-name')
blob_container = dbutils.secrets.get(scope='gascaribe', key='bs-container')
blob_storage = storage_account_name + ".blob.core.windows.net"
config_key = "fs.azure.account.key."+storage_account_name+".blob.core.windows.net"
blob_access_key = dbutils.secrets.get(scope='gascaribe', key='bs-access-key')
spark.conf.set(config_key, blob_access_key)

# COMMAND ----------

query = 'SELECT * FROM ComercializacionML.DatosEDA'

# COMMAND ----------

df = spark.read \
  .format("com.databricks.spark.sqldw") \
  .option("url", sqlDwUrl) \
  .option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
  .option("forwardSparkAzureStorageCredentials", "true") \
  .option("maxStrLength", "1024" ) \
  .option("query", query) \
  .load()

rawData = df.toPandas()

# COMMAND ----------

ordinalNombre = {}
ordinalTipoUsuario = {}
for i,est in enumerate(rawData['Nombre'].unique()):
    ordinalNombre[est] = i

for i,tipo in enumerate(rawData['TipoUsuario'].unique()):
    ordinalTipoUsuario[tipo] = i


ordinalTipoUsuario

# COMMAND ----------

def process_inputs(df,ordinalNombre=ordinalNombre,ordinalTipoUsuario=ordinalTipoUsuario):
    df = df.copy()
    
    ## Filtrar por tipo de usuario
    #df = df[df['TipoUsuario'] == tipoUsuario]
    
    festivosBin = []
    for fecha in df['Fecha']:
        if str(fecha) in festivos:
            festivosBin.append(1)
        else:
            festivosBin.append(0)
            
    df['Festivos'] = festivosBin
    
    festivosBinEspeciales = []
    for fecha in df['Fecha']:
        if str(fecha) in festivosEspeciales:
            festivosBinEspeciales.append(1)
        else:
            festivosBinEspeciales.append(0)
            
    df['FestivosEspeciales'] = festivosBinEspeciales
    
    # Cambiar fecha por datetime
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['DiaDeSemana'] = df['Fecha'].apply(lambda x: x.dayofweek)

    for i,festivo in enumerate(df['Festivos']):
        if festivo == 1:
            df['DiaDeSemana'][i] = 8
        else:
            pass

    df['OrdinalNombre'] = df['Nombre'].replace(ordinalNombre).astype(int)
    df['OrdinalTipoUsuario'] = df['TipoUsuario'].replace(ordinalTipoUsuario).astype(int)
    
    # Reemplazar valores que salen como 0E-8 a 0.
    df['VolumenM3'] = df['VolumenM3'].replace({0E-8:0})
    
    
    uniqueDispositivos = df['IdDispositivo'].unique()
    
    newDisp = pd.DataFrame(columns=['IdDispositivo', 'Nombre', 'TipoUsuario', 'Fecha', 'VolumenM3','VolumenCumSum','DiaDeSemana'])
    
    for disp in uniqueDispositivos:
        dfDispSum = df[df['IdDispositivo'] == disp].sort_values(by='Fecha')
        try:
            dfDispSum['VolumenCumSum'] = dfDispSum['VolumenM3'].cumsum().astype('float')
        except Exception as e:
            print(f'{disp}: {e}')
        
        newDisp = pd.concat([dfDispSum,newDisp],axis=0)
    
    newDisp['VolumenM3'] = newDisp['VolumenM3'].astype('float')
    newDisp['Festivos'] = newDisp['Festivos'].astype('int')
    newDisp['FestivosEspeciales'] = newDisp['FestivosEspeciales'].astype('int')
    

    return newDisp
    

# COMMAND ----------

X = process_inputs(rawData)

# COMMAND ----------

X.head()

# COMMAND ----------

len(X[X['VolumenM3'] > 0])/len(X)

# COMMAND ----------

import pandas as pd
from scipy.stats import ttest_ind

def compareHolidays(df, consumo, day_column, holiday_column,TipoUsuario,soloConDiasDeSemana=True):
    df = df.dropna()
    # Filter data for holidays and normal days
    holidays = df[(df[holiday_column] == 1) & (df['TipoUsuario'] == TipoUsuario)][consumo]
    if soloConDiasDeSemana:
        normal_days = df[(df[day_column].isin([1,2,3,4,5])) & (df['TipoUsuario'] == TipoUsuario)][consumo]
    else:
        normal_days = df[(df[holiday_column] == 0) & (df['TipoUsuario'] == TipoUsuario)][consumo]
    
    # t-test
    t_statistic, p_value = ttest_ind(holidays, normal_days)
    
    # Print the results
    print("Comparacion de consumo de gas:")
    print("----------------------------")
    print("Media de Consumo de Gas en Dia Festivo: {:.2f} m3".format(holidays.mean()))
    print("Media de Consumo de Gas en Dia Normal: {:.2f} m3".format(normal_days.mean()))
    print("T-Statistic: {:.2f}".format(t_statistic))
    print("p-Value: {:.4f}".format(p_value))
    
    # Interpret the results
    if p_value < 0.05:
        print("Hay diferencia estadisticamente significativa entre los dias festivos y los normales.")
    else:
        print("No hay diferencia estadisticamente significativa entre los dias festivos y los normales.")


# COMMAND ----------

for usuario in X['TipoUsuario'].unique():
    print(usuario)
    compareHolidays(X,'VolumenM3','DiaDeSemana','Festivos',usuario,soloConDiasDeSemana=True)

# COMMAND ----------

X[X['VolumenM3'].isna()].reset_index(drop=True).iloc[0]['DiaDeSemana']

# COMMAND ----------

xd = X[X['VolumenM3'].isna()]

#xd[(xd['Nombre'] == 'SUAN') & (xd['Festivos'] == 1) & (xd[xd['DiaDeSemana'] == 2])]
xd.shape

# COMMAND ----------

def replaceNaN(df,festivos=True):
    df = df.copy()

    nanValues = df[df['VolumenM3'].isna()].reset_index(drop=True)
    nonnanValues = df[~df['VolumenM3'].isna()].reset_index(drop=True)

    for row,value in enumerate(nanValues['VolumenM3']):
        diaDeSemana = nanValues['DiaDeSemana'].iloc[row]
        estacion = nanValues['Nombre'].iloc[row]
        festivo = nanValues['Festivos'].iloc[row]
        if festivo:
            listOfValues = list(nonnanValues[(nonnanValues['Nombre'] == estacion) & (nonnanValues['Festivos'] == 1)].tail(4)['VolumenM3'])
            replaceValue = sum(listOfValues)/len(listOfValues)
        else:
            listOfValues = list(nonnanValues[(nonnanValues['Nombre'] == estacion) & (nonnanValues['Festivos'] == 0) & (nonnanValues['DiaDeSemana'] == diaDeSemana)].tail(4)['VolumenM3'])
            replaceValue = sum(listOfValues)/len(listOfValues)
        
        nanValues['VolumenM3'].iloc[row] = replaceValue
    

    newdf = pd.concat([nanValues,nonnanValues],axis=0)
    newdf = newdf.sort_values(by=['Nombre', 'Fecha'])

    return newdf

# COMMAND ----------

def replaceOutliers(df,festivos=True):
    df = df.copy()

    for est in df['Nombre'].unique():
        estacion = df[df['Nombre'] == est].reset_index(drop=True)
        

    nanValues = df[df['VolumenM3'].isna()].reset_index(drop=True)
    nonnanValues = df[~df['VolumenM3'].isna()].reset_index(drop=True)

    for row,value in enumerate(nanValues['VolumenM3']):
        diaDeSemana = nanValues['DiaDeSemana'].iloc[row]
        estacion = nanValues['Nombre'].iloc[row]
        festivo = nanValues['Festivos'].iloc[row]
        if festivo:
            listOfValues = list(nonnanValues[(nonnanValues['Nombre'] == estacion) & (nonnanValues['Festivos'] == 1)].tail(4)['VolumenM3'])
            replaceValue = sum(listOfValues)/len(listOfValues)
        else:
            listOfValues = list(nonnanValues[(nonnanValues['Nombre'] == estacion) & (nonnanValues['Festivos'] == 0) & (nonnanValues['DiaDeSemana'] == diaDeSemana)].tail(4)['VolumenM3'])
            replaceValue = sum(listOfValues)/len(listOfValues)
        
        nanValues['VolumenM3'].iloc[row] = replaceValue
    

    newdf = pd.concat([nanValues,nonnanValues],axis=0)
    newdf = newdf.sort_values(by=['Nombre', 'Fecha'])

    return newdf

# COMMAND ----------

import pandas as pd
from scipy.stats import ttest_ind

def compareHolidays(df, consumo, day_column, holiday_column,TipoUsuario,dia):
    df = df.dropna()
    # Filter data for holidays and normal days
    holidays = df[(df[holiday_column] == 1) & (df['TipoUsuario'] == TipoUsuario)][consumo]
    
    normal_days = df[(df['FestivosEspeciales'] == 1) & (df['TipoUsuario'] == TipoUsuario)][consumo]
    
    # t-test
    t_statistic, p_value = ttest_ind(holidays, normal_days)
    
    # Print the results
    print("Comparacion de consumo de gas:")  
    print("----------------------------")
    print("Media de Consumo de Gas en Dia Festivo: {:.2f} m3".format(holidays.mean()))
    print("Media de Consumo de Gas en Dia Normal: {:.2f} m3".format(normal_days.mean()))
    print("T-Statistic: {:.2f}".format(t_statistic))
    print("p-Value: {:.4f}".format(p_value))
    
    # Interpret the results
    if p_value < 0.05:
        print("Hay diferencia estadisticamente significativa entre los dias festivos y los normales.")
    else:
        print("No hay diferencia estadisticamente significativa entre los dias festivos y los normales.")


# COMMAND ----------

for usuario in X['TipoUsuario'].unique():
    print(usuario)
    compareHolidays(X,'VolumenM3','DiaDeSemana','Festivos',usuario,dia=0)

# COMMAND ----------

for usuario in X['TipoUsuario'].unique():
    print(usuario)
    compareHolidays(X,'VolumenM3','DiaDeSemana','FestivosEspeciales',usuario,dia=6)

# COMMAND ----------

X[X['DiaDeSemana'] == 0]

# COMMAND ----------


