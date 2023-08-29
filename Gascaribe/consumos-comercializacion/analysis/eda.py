# Databricks notebook source
import os
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, DateType
from datetime import date,datetime
today = datetime.now()
today_dt = today.strftime("%d-%m-%Y")

import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

# COMMAND ----------

dwDatabase = os.environ.get("DWH_NAME_GDC")
dwServer = os.environ.get("DWH_HOST_GDC")
dwUser = os.environ.get("DWH_USER_GDC")
dwPass = os.environ.get("DWH_PASS_GDC")
dwJdbcPort = os.environ.get("DWH_PORT_GDC")
dwJdbcExtraOptions = ""
sqlDwUrl = "jdbc:sqlserver://" + dwServer + ".database.windows.net:" + dwJdbcPort + ";database=" + dwDatabase + ";user=" + dwUser + ";password=" + dwPass + ";" + dwJdbcExtraOptions
storage_account_name = os.environ.get("BS_NAME_GDC")
blob_container = os.environ.get("BS_CONTAINER_GDC")
blob_storage = storage_account_name + ".blob.core.windows.net"
config_key = "fs.azure.account.key."+storage_account_name+".blob.core.windows.net"
blob_access_key = os.environ.get("BS_ACCESS_KEY_GDC")
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

rawData['TipoUsuario'].value_counts()

# COMMAND ----------

estacion = rawData[rawData['TipoUsuario'] == 'ESTACION']
estacion['IdDispositivo'].value_counts()

# COMMAND ----------

def process_inputs(df,tipoUsuario):
    df = df.copy()
    
    # Filtrar por tipo de usuario
    df = df[df['TipoUsuario'] == tipoUsuario]
    
    # Cambiar fecha por datetime
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # Reemplazar valores que salen como 0E-8 a 0.
    df['VolumenM3'] = df['VolumenM3'].replace({0E-8:0})
    
    
    uniqueDispositivos = df['IdDispositivo'].unique()
    
    newDisp = pd.DataFrame(columns=['IdDispositivo', 'Nombre', 'TipoUsuario', 'Red', 'Fecha', 'VolumenM3','VolumenCumSum'])
    
    for disp in uniqueDispositivos:
        dfDispSum = df[df['IdDispositivo'] == disp].sort_values(by='Fecha')
        try:
            dfDispSum['VolumenCumSum'] = dfDispSum['VolumenM3'].cumsum().astype('float')
        except:
            print(disp)
        
        newDisp = pd.concat([dfDispSum,newDisp],axis=0)
    
    newDisp['VolumenM3'] = newDisp['VolumenM3'].astype('float')
    
    
    
    
    return newDisp
    

# COMMAND ----------

X = process_inputs(rawData,'ESTACION')
X.info()

# COMMAND ----------

X.head()

# COMMAND ----------

X.groupby('IdDispositivo').agg({'VolumenM3':['count']})

# COMMAND ----------

X[X['IdDispositivo'] == '4-354-1-2'].sort_values(by='Fecha')['VolumenM3']

# COMMAND ----------

uniqueDispositivos = X['IdDispositivo'].unique()

# COMMAND ----------

quantile50 = X['VolumenCumSum'].quantile(0.5)
quantile80 = X['VolumenCumSum'].quantile(0.8)
quantile95 = X['VolumenCumSum'].quantile(0.95)
quantile99 = X['VolumenCumSum'].quantile(0.99)

# COMMAND ----------

import matplotlib

# COMMAND ----------

fig, ax = plt.subplots(figsize=(20,10))
counter = 0
for disp in uniqueDispositivos:
    df = X[X['IdDispositivo'] == disp].sort_values(by='Fecha')
    try:
        df['cumsum'] = df['VolumenM3'].cumsum()

        plt.plot(df['Fecha'],df['cumsum'])
    except:
        print(disp)
    counter += 1
    
plt.ylabel('Consumo acumulado (m3)')
plt.xlabel('Fecha de consumo')
plt.axhline(y=quantile50,color='#dc143c',linestyle='dotted')
plt.axhline(y=quantile80,color='#dc143c',linestyle='dotted')
plt.axhline(y=quantile95,color='#dc143c',linestyle='dotted')
plt.axhline(y=quantile99,color='#dc143c',linestyle='dotted')

ax.axvspan(datetime(2020, 1, 1), datetime(2021, 1, 1), alpha=0.5, color='#ffcccb')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
print(counter)
plt.show()
    
    

# COMMAND ----------

gb = X.groupby('Nombre').agg({'VolumenCumSum':['max']})
gb.columns = ['_'.join(col) for col in gb.columns.values]
top10 = gb.sort_values(by='VolumenCumSum_max',ascending=False).head(10)

# COMMAND ----------

fig, ax = plt.subplots(figsize=(20,10))
ax.bar(x=top10.index,height=top10['VolumenCumSum_max'],color=['#86bb76','#dfe6e7','#1d4893','#8897bf','#94bfcc','#526ca8','#6c7cb4','#042484','#6198b5'])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.ylabel('Consumo acumulado (m3)')
plt.xlabel('Estacion')
plt.xticks(rotation=45)
plt.title('Consumo acumulado de estaciones')
plt.show()

# COMMAND ----------

print(X[X['Nombre'] == 'KM0 CIENAGA - STA MARTA (MED 1)']['VolumenCumSum'].tail(1).values[0] == X[X['Nombre'] == 'KM0 CIENAGA - STA MARTA (MED 2)']['VolumenCumSum'].tail(1).values[0])

# COMMAND ----------

X['Mes'] = X['Fecha'].apply(lambda x: x.month)
X['Ano'] = X['Fecha'].apply(lambda x: x.year)
X['MesAno'] = X['Mes'].astype(str) + '-' + X['Ano'].astype(str)
X['DayofWeek'] = X['Fecha'].apply(lambda x: x.day_name())
X['Month'] = X['Fecha'].apply(lambda x: x.month_name())
X['DayofWeek'] = X['DayofWeek'].replace({
                                            'Monday':'Lunes',
                                            'Tuesday':'Martes',
                                            'Wednesday':'Miercoles',
                                            'Thursday':'Jueves',
                                            'Friday':'Viernes',
                                            'Saturday':'Sabado',
                                            'Sunday':'Domingo'
})
X['Month'] = X['Month'].replace({
                                            'January':'Enero',
                                            'February':'Febrero',
                                            'March':'Marzo',
                                            'April':'Abril',
                                            'May':'Mayo',
                                            'June':'Junio',
                                            'July':'Julio',
                                            'August':'Agosto',
                                            'September':'Septiembre',
                                            'October':'Octubre',
                                            'November':'Noviembre',
                                            'December':'Diciembre'
                                        })

# COMMAND ----------

X.head()

# COMMAND ----------

gb = X.groupby(['MesAno']).agg({'VolumenM3':['sum']})
gb.columns = ['_'.join(col) for col in gb.columns.values]
top20 = gb.sort_values(by='VolumenM3_sum',ascending=False).head(50).reset_index()

# COMMAND ----------

fig, ax = plt.subplots(figsize=(25,10))
ax.bar(x=top20['MesAno'],height=top20['VolumenM3_sum'],color=['#86bb76','#dfe6e7','#1d4893','#8897bf','#94bfcc','#526ca8','#6c7cb4','#042484','#6198b5'])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.ylabel('Consumo acumulado (m3)')
plt.xlabel('Mes-Ano')
plt.xticks(rotation=45)
plt.title('Consumo acumulado por mes-ano')
plt.show()

# COMMAND ----------

gb = X.groupby(['Month']).agg({'VolumenM3':['sum']})
gb.columns = ['_'.join(col) for col in gb.columns.values]
top12 = gb.sort_values(by='VolumenM3_sum',ascending=False).reset_index()

fig, ax = plt.subplots(figsize=(25,10))
ax.bar(x=top12['Month'],height=top12['VolumenM3_sum'],color=['#86bb76','#dfe6e7','#1d4893','#8897bf','#94bfcc','#526ca8','#6c7cb4','#042484','#6198b5'])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.ylabel('Consumo acumulado (m3)')
plt.xlabel('Mes')
plt.xticks(rotation=45)
plt.title('Consumo acumulado por mes')
plt.show()

# COMMAND ----------

def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{:.1f}%\n({v:d})'.format(pct, v=val)
    return my_format


gb = X[(X['Ano'] != 2023) & (X['Ano'] != 2020)].groupby(['Ano']).agg({'VolumenM3':['sum']}).reset_index()

gb.columns = ['_'.join(col) for col in gb.columns.values]
top12 = gb.sort_values(by='VolumenM3_sum',ascending=False).reset_index()

fig, ax = plt.subplots(figsize=(25,10))
#ax.pie(top12['VolumenM3_sum'],colors=['#86bb76','#dfe6e7','#1d4893','#8897bf','#94bfcc','#526ca8','#6c7cb4','#042484','#6198b5'])
ax.pie(top12['VolumenM3_sum'],
        colors=['#86bb76','#dfe6e7','#1d4893','#8897bf','#94bfcc','#526ca8','#6c7cb4','#042484','#6198b5'],
        labels=top12['Ano_'],
        autopct=autopct_format(top12['VolumenM3_sum']))

plt.title('Consumo acumulado en m3 por ano')
plt.show()

# COMMAND ----------

gb = X.groupby('IdDispositivo').agg({'VolumenM3':['sum']})
gb.columns = ['_'.join(col) for col in gb.columns.values]
low = gb.reset_index().sort_values(by='VolumenM3_sum').head()


high = gb.reset_index().sort_values(by='VolumenM3_sum',ascending=False).head()

highlow = pd.concat([low,high],axis=0)
highlow

# COMMAND ----------

fig, ax = plt.subplots(figsize=(20,10))
plt.figure(figsize=(25,15))
for disp in highlow['IdDispositivo']:
    df = X[X['IdDispositivo'] == disp].sort_values(by='Fecha')
    try:
        df['cumsum'] = df['VolumenM3'].cumsum()

        ax.plot(df['Fecha'],df['cumsum'])
    except:
        print(disp)
ax.axvspan(datetime(2020, 1, 1), datetime(2021, 1, 1), alpha=0.5, color='#ffcccb')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.ylabel('Consumo acumulado (m3)')
plt.xlabel('Fecha de consumo')

plt.show()

# COMMAND ----------

| newX[newX['VolumenCumSum'] == newX['VolumenCumSum'].max()]

# COMMAND ----------



# COMMAND ----------

X.columns

# COMMAND ----------

newX = pd.DataFrame(columns=X.columns)


# COMMAND ----------

festivos2018 = ['']
