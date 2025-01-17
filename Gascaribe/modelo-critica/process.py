# Databricks notebook source
import os as os
import pandas as pd
import numpy as np
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, DateType
from datetime import date,datetime
today = datetime.now()
today_dt = today.strftime("%Y-%m-%d")

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

from delta.tables import DeltaTable
from pyspark.sql.functions import *
# Load the Delta table
delta_table = DeltaTable.forName(spark,"bigdc.modelocritica.basecritica")

# Read the data into a DataFrame
data = delta_table.toDF()
rawData = data.toPandas()
rawdata=rawData.copy()

# COMMAND ----------

rawdata['FechaUltimaLecturaValida'] = rawdata['FechaUltimaLecturaValida'].fillna('1900-01-01')

# COMMAND ----------

rawdata['FechaUltimaLecturaValida'] = pd.to_datetime(rawdata['FechaUltimaLecturaValida'])
rawdata['FechaEjecucion'] = pd.to_datetime(rawdata['FechaEjecucion'])
rawdata['FechaLectura']=pd.to_datetime(rawdata['FechaLectura'])

# COMMAND ----------

# Fill NAs
rawdata['IdTipoTrabajo'].fillna(0.0,inplace=True)
rawdata['UltimaLecturaValida'].fillna('0.0',inplace=True)
rawdata['ValorLecturaValido0'].fillna(0,inplace=True)
rawdata['UltimaCausal'].fillna(0,inplace=True)
rawdata['ConsumoPromedioLocalidadSubCategoria'].fillna('0.0',inplace=True)
rawdata['ValorLectura'].fillna('-1000.0',inplace=True)
rawdata['ValorLecturaAnterior'].fillna('-1000.0',inplace=True)
rawdata['LecturaDosMeses'].fillna('-1000.0',inplace=True)
rawdata['ConsumoPromedioProducto'].fillna('-100.0',inplace=True)

# COMMAND ----------

# Check decimals with , instead of .
try:
    rawdata['LecturaDosMeses']=rawdata['LecturaDosMeses'].apply(lambda x: x.replace(',','.'))
except: 
    pass
  
try:
    rawdata['ValorLecturaAnterior']=rawdata['ValorLecturaAnterior'].apply(lambda x: x.replace(',','.'))
except: 
    pass

# Check decimals with , instead of .
try:
    rawdata['ConsumoPromedioLocalidadSubCategoria']=rawdata['ConsumoPromedioLocalidadSubCategoria'].apply(lambda x: x.replace(',','.'))
except: 
    pass

# Check decimals with , instead of .
try:
    rawdata['ConsumoEstimadoProducto']=rawdata['ConsumoEstimadoProducto'].apply(lambda x: x.replace(',','.'))
except: 
    pass

# Check decimals with , instead of .
try:
    rawdata['ComentarioOrden']=rawdata['ComentarioOrden'].apply(lambda x: x.replace('/', ' , '))
except: 
    pass
try:
    rawdata['Lectura12620']=rawdata['Lectura12620'].apply(lambda x: x.replace(',', '.'))
except: 
    pass
try:
    rawdata['ConsumoPromedioProducto']=rawdata['ConsumoPromedioProducto'].apply(lambda x: x.replace(',', '.'))
except: 
    pass
try:
    rawdata['UltimaLecturaValida']=rawdata['UltimaLecturaValida'].apply(lambda x: x.replace(',', '.'))
except: 
    pass

# COMMAND ----------

# String to numeric types
rawdata['IdTipoTrabajo']=rawdata['IdTipoTrabajo'].astype('int64')
rawdata['VolumenActual']=rawdata['VolumenActual'].astype('float')
rawdata['ConsumoPromedioProducto']=rawdata['ConsumoPromedioProducto'].astype('float')
rawdata['ConsumoPromedioLocalidadSubCategoria']=rawdata['ConsumoPromedioLocalidadSubCategoria'].astype('float')
rawdata['ConsumoEstimadoProducto']=rawdata['ConsumoEstimadoProducto'].astype('float')
rawdata['ValorLecturaAnterior']=rawdata['ValorLecturaAnterior'].astype('float')
rawdata['ValorLectura']=rawdata['ValorLectura'].astype('float')
rawdata['LecturaDosMeses']=rawdata['LecturaDosMeses'].astype('float')
rawdata['UltimaLecturaValida']=rawdata['UltimaLecturaValida'].astype('float')
rawdata['ValorLecturaValido0']=rawdata['ValorLecturaValido0'].astype('float')
rawdata['LecturaUltimaOrden']=rawdata['LecturaUltimaOrden'].astype('float')


# COMMAND ----------

rawdata['Regla 0']=rawdata[['UltimaLecturaValida','ValorLecturaValido0','LecturaDosMeses']].apply(lambda x: 0 if x[0]==-10 else
                                                                                        1 if x[0]<=x[1] else 0,axis=1)

# COMMAND ----------

rawdata['RI'] = rawdata['RI'].fillna(-300)
rawdata['RS'] = rawdata['RS'].fillna(300)
rawdata['DiferenciaPromedioPeriodo']=rawdata[['VolumenActual','ConsumoPromedioProducto']].apply(lambda x: (x[0]-x[1])/x[1] if x[0]<> 0 
                                                                                                            else x[0]
                                                                                                ,axis=1)

# COMMAND ----------

rawdata['Regla -1']=rawdata[['DiferenciaPromedioPeriodo','RI','RS','ConsumoPromedioProducto','FechaUltimaLecturaValida','FechaLectura']].apply(lambda x: 0 if x[0]>x[2] else
                                                                                     0 if x[0]<x[1] 
                                                                                     else 1,axis=1)

# COMMAND ----------

#Primero verificaremos que exista una orden existosa que pueda justificar el consumo

rawdata['Regla1.1'] = rawdata[['IdTipoTrabajo','UltimaCausal']].apply(lambda x: 'SI' if x[0] in (12620,12526,10546,12521,12137,12190,12187,12690,12527,10559,12688,12689)    else 'NO', axis=1)
rawdata['Regla1.3']=rawdata[['FechaLectura','FechaEjecucion','LecturaUltimaOrden','ValorLectura']].apply(lambda x: 'NO' if x[1]=='1900-01-01' 
                                                                                                            else 'SI' if (x[0]<=x[1] and x[3]<=x[2]) or (x[0]>=x[1] and x[3]>=x[2])
                                                                                                            else 'NO',axis=1 )
rawdata['Regla1.2']=rawdata[['FechaLectura','FechaEjecucion','LecturaUltimaOrden','ValorLectura','UltimaLecturaValida','FechaUltimaLecturaValida']].apply(lambda x: 'NO' if x[1]=='01-01-1900' or x[5]=='01-01-1900'
                                                    else 'SI' if x[1]>x[5] and x[0]>x[5] and (x[2]+((x[2]-x[4])/((x[1]-x[5]).days))*((x[0]-x[5]).days)) >=x[3]
                                                                                                            else 'NO',axis=1 )

rawdata['Regla 1'] = rawdata[['Regla1.1','Regla1.3','Regla 0','LecturaUltimaOrden','LecturaDosMeses','Regla1.2']].apply(lambda x: 'NO' if x[3]==-10 else
'NO' if x[4]>x[3] else
'Yes' if x[2]==1 and (x[0]=='SI' and x[1]=='SI' and x[5]=='SI' ) else 'NO',axis=1)
rawdata.pop('Regla1.1')
rawdata.pop('Regla1.3')
rawdata.pop('Regla1.2')
#modificar regla 1. Para empezar el campo de lectura de la orden debe no ser nulo. En caso dado sea nulo, verificar que en al menos una de las ordenes asignadas dentro del periodo de consumo exista la campo de lectura y tomar esa orden.
#lo segundo que tengas las causales indicadas a los tt indicados ES UN Y
# dejar de verificar los comentarios 
#Regla 1: tiene que tener lectura. Si tiene la causal error de lectura 3194 mandar a revisión en seguida. Lectura debe ser coherente (dependiendo de la fecha de ejecución)
#datos adicionales 12620 10944 12688 12689 12486 12690 todos deben tener campo de lectura, los tipos de trabajo se verifica como se venía haciendo y si no está no está
#realizar metabase que indique los productos que todas las ordenes legalizadas en el periodo de consumo no tengan el campo de lectura.

# COMMAND ----------

#Regla numero 2, se verifica si la categoria es 1 o 2 y si el producto estuvo suspendido en algún momento durante el periodo de facturacion y se verifica la diferencia que puedan estar en el rango permitido 
rawdata['Regla 2.1'] =rawdata[['ConsumoEstimadoProducto','ConsumoPromedioProducto','Regla 1','ConsumoPromedioLocalidadSubCategoria','Regla 0','RI','RS']].apply(lambda x: -12487531478 if  x[2]=='Yes' else 
                                                                                                              ( x[0] - x[1] )/ x[1] * 100 if x[2]=='NO' and x[1]!=0 else
                                                                                                              ( x[0] - x[3] )/ x[3] * 100 if x[1]==0 and x[2]=='NO' and x[3]!=0 
                                                                                                              else 1000,axis=1)
rawdata['Regla 2']= rawdata[['Regla 1','Regla 2.1','RI','RS','Regla 0','ConsumoPromedioProducto','DiasPosiblesSuspendidos','VolumenActual','DiferenciaPromedioPeriodo']].apply(lambda x: 
                                                                  'Aplica Regla 1' if x[0]=='Yes'   
                                                                    else 'NO' if x[6]==0 
                                                                    else 'NO' if x[6]>=0 and (x[1]<x[2] or x[1]>x[3])
                                                                    else 'Yes' if x[7]!=0 and x[1]<=x[3] and x[1]>=x[2] and x[4]==1
                                                                    else 'NO',axis=1)
rawdata.pop('Regla 2.1')

# COMMAND ----------

#Regla 4 Verificar la desviacion del consumo con respecto a la subcategoria directamente 
rawdata['Regla 4'] = rawdata[['Regla 1','Regla 2','ConsumoPromedioLocalidadSubCategoria','VolumenActual','Regla 0','IdCategoria','ConsumoPromedioProducto','Regla -1']].apply(lambda x: 
                                                                                                         'Aplica Regla 1' if x[0]=='Yes'
                                                                                                    else 'Aplica Regla 2' if x[1]=='Yes'
                                                                                                    else 'Aplica Regla -1' if x[7]==1
                                                                                                    else 'NO' if x[5]==2 or x[2]==0 
                                                                                                    else 'NO' if x[3]>x[2]
                                else 'Yes' if x[3]>0 and x[3]<x[2] #(100*(x[3]-x[2])/x[2])<=x[5] and (100*(x[3]-x[2])/x[2])>=x[4] and x[6]==1 and x[2]!=0  
                                          else 'No',axis=1)
# caso comerciales: no se debe utilizar la regla 4 

# COMMAND ----------

rawdata['FechaPrediccion'] = today_dt
rawdata['FechaPrediccion'] = pd.to_datetime(rawdata['FechaPrediccion'])
dfNew =rawdata[['IdOrden','Idproducto','IdCiclo','PeriodoConsumo','VolumenActual','Regla 4','FechaPrediccion','Regla -1']]

# COMMAND ----------

schema = StructType([
    StructField("IdOrden", IntegerType(), True),
    StructField("Idproducto", IntegerType(), True),
    StructField("IdCiclo", IntegerType(), True),
    StructField("PeriodoConsumo", IntegerType(), True),
    StructField("VolumenActual", FloatType(), True),
    StructField("Regla4", StringType(), True),
    StructField("FechaPrediccion", DateType(), True)
    ])
df = spark.createDataFrame(dfNew, schema = schema)
df.write \
.format("com.databricks.spark.sqldw") \
.option("url", sqlDwUrl) \
.option("forwardSparkAzureStorageCredentials", "true") \
.option("dbTable", "ModeloCritica.StageProcesado") \
.option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
.mode("overwrite") \
.save()
