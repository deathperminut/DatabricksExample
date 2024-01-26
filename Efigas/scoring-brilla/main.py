# Databricks notebook source
import pandas as pd
import numpy as np
import os
from azure.storage.blob import *
import io
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, DateType
from datetime import datetime, date
from delta.tables import *
from pyspark.sql.functions import *

# COMMAND ----------

def preprocess_inputs(df):

    df = df.copy()
    df = df[~df['Identificacion'].isna() & df['Categoria']==1]
    df = df.rename(columns={'SubCategoria':'Estrato'})
 
    return df

# COMMAND ----------

def arbol(df):
    node_Brilla = []
    riesgo_Brilla = []
    node_GDC = []
    riesgo_GDC = []
    for i in range(0, len(df)):
        # Brilla
        if (df['Edad60Brilla'].iloc[i] <= 0) and (df['Edad0Brilla'].iloc[i] > 8) and (df['CuotasPendientesBrilla'].iloc[i] <= 7 or np.isnan(df['CuotasPendientesBrilla'].iloc[i])) and (df['Edad30Brilla'].iloc[i] <= 0):
            node_Brilla.append(1)
            riesgo_Brilla.append('bajo')

        elif (df['Edad60Brilla'].iloc[i] <= 0) and (df['Edad0Brilla'].iloc[i] > 8) and (df['CuotasPendientesBrilla'].iloc[i] <= 7 or np.isnan(df['CuotasPendientesBrilla'].iloc[i])) and (df['Edad30Brilla'].iloc[i] > 0):
            node_Brilla.append(2)
            riesgo_Brilla.append('bajo')

        elif (df['Edad60Brilla'].iloc[i] <= 0) and (df['Edad0Brilla'].iloc[i] > 8) and (df['CuotasPendientesBrilla'].iloc[i] > 7):
            node_Brilla.append(3)
            riesgo_Brilla.append('medio')  # Bajo -> Medio V.2

        elif (df['Edad60Brilla'].iloc[i] <= 0) and (df['Edad0Brilla'].iloc[i] > 1 and df['Edad0Brilla'].iloc[i] <= 8) and (df['CuotasPendientesBrilla'].iloc[i] <= 34 or np.isnan(df['CuotasPendientesBrilla'].iloc[i])):
            node_Brilla.append(4)
            riesgo_Brilla.append('medio')

        elif (df['Edad60Brilla'].iloc[i] <= 0) and (df['Edad0Brilla'].iloc[i] > 1 and df['Edad0Brilla'].iloc[i] <= 8) and (df['CuotasPendientesBrilla'].iloc[i] > 34):
            node_Brilla.append(5)
            riesgo_Brilla.append('medio')

        elif (df['Edad60Brilla'].iloc[i] <= 0) and (df['Edad0Brilla'].iloc[i] <= 1):
            node_Brilla.append(6)
            riesgo_Brilla.append('medio')

        elif (df['Edad60Brilla'].iloc[i] > 0):
            node_Brilla.append(7)
            riesgo_Brilla.append('alto')

        else:
            node_Brilla.append('missing')
            riesgo_Brilla.append('missing')

        # Gas
        if (df['Comport_Pago_Gas'].iloc[i] == '1-Pago A Tiempo') and (df['Edad0Gas'].iloc[i] > 12) and (df['CuotasPendientesGas'].iloc[i] <= 35):
            node_GDC.append(8)
            riesgo_GDC.append('medio')  # Bajo -> Medio V.4

        elif (df['Comport_Pago_Gas'].iloc[i] == '1-Pago A Tiempo') and (df['Edad0Gas'].iloc[i] > 12) and (df['CuotasPendientesGas'].iloc[i] > 35 or np.isnan(df['CuotasPendientesGas'].iloc[i])):
            node_GDC.append(9)
            riesgo_GDC.append('medio')  # Bajo -> Medio V.4

        elif (df['Comport_Pago_Gas'].iloc[i] == '1-Pago A Tiempo') and (df['Edad0Gas'].iloc[i] > 8 and df['Edad0Gas'].iloc[i] <= 12) and (df['CuotasPendientesGas'].iloc[i] <= 35):
            node_GDC.append(10)
            riesgo_GDC.append('medio')  # Bajo -> Medio V.4

        elif (df['Comport_Pago_Gas'].iloc[i] == '1-Pago A Tiempo') and (df['Edad0Gas'].iloc[i] <= 8) and (df['CuotasPendientesGas'].iloc[i] <= 35):
            node_GDC.append(11)
            riesgo_GDC.append('medio')  # Bajo -> Medio V.4

        elif (df['Comport_Pago_Gas'].iloc[i] == '1-Pago A Tiempo') and (df['Edad0Gas'].iloc[i] > 8 and df['Edad0Gas'].iloc[i] <= 12) and (df['CuotasPendientesGas'].iloc[i] > 35 or np.isnan(df['CuotasPendientesGas'].iloc[i])):
            node_GDC.append(12)
            riesgo_GDC.append('medio')  # Bajo -> Medio V.2

        elif (df['Comport_Pago_Gas'].iloc[i] == '1-Pago A Tiempo') and (df['Edad0Gas'].iloc[i] <= 8) and (df['CuotasPendientesGas'].iloc[i] > 35 or np.isnan(df['CuotasPendientesGas'].iloc[i])):
            node_GDC.append(13)
            riesgo_GDC.append('medio')

        elif (df['Comport_Pago_Gas'].iloc[i] == '2-Pago Atrasado a 60 Dias') and (df['Edad60Gas'].iloc[i] <= 1):
            node_GDC.append(14)
            riesgo_GDC.append('medio')

        elif (df['Comport_Pago_Gas'].iloc[i] == '2-Pago Atrasado a 60 Dias') and (df['Edad60Gas'].iloc[i] > 1):
            node_GDC.append(15)
            riesgo_GDC.append('medio')

        elif (df['Comport_Pago_Gas'].iloc[i] == '3-No Paga a 90 Dias') and (df['EdadM90Gas'].iloc[i] == 0):
            node_GDC.append(16)
            riesgo_GDC.append('medio')

        elif (df['Comport_Pago_Gas'].iloc[i] == '3-No Paga a 90 Dias'):
            node_GDC.append(17)
            riesgo_GDC.append('alto')

        else:
            node_GDC.append('missing')
            riesgo_GDC.append('missing')

    df['Nodo Brilla'] = node_Brilla
    df['Riesgo Brilla'] = riesgo_Brilla
    df['Nodo Gas'] = node_GDC
    df['Riesgo Gas'] = riesgo_GDC

    return df

# COMMAND ----------

def riesgo_total(df):
    riesgo_combinado = []
    for i in range(len(df)):
        if df['Riesgo Brilla'].iloc[i] != 'missing':
            riesgo_combinado.append(df['Riesgo Brilla'].iloc[i])
        else:
            riesgo_combinado.append(df['Riesgo Gas'].iloc[i])

    df['Riesgo Combinado'] = riesgo_combinado

    return df

# COMMAND ----------

def riesgo_codificado(estrato, riesgo, brilla):
    r = 0
    if(riesgo == 'bajo'):
        r = 1
    
    b = 0
    if(brilla != 'missing'):
        b = 1

    return estrato * 100 + r * 10 + b

# COMMAND ----------

def nodos(df):

    nuevo = []

    for i in range(len(df)):
        if df['Nodo Brilla'].iloc[i] != 'missing':
            nuevo.append(int(df['Nodo Brilla'].iloc[i]))
        elif df['Nodo Brilla'].iloc[i] == 'missing':
            nuevo.append(int(df['Nodo Gas'].iloc[i]))

    df['Nodo Combinado'] = nuevo

    df['Nodo codificado'] = df.apply(lambda x : riesgo_codificado(x['Estrato'], x['Riesgo Combinado'], x['Riesgo Brilla']), axis=1)

    return df

# COMMAND ----------

def cupos(df):

    df = df.copy()

    cupos_efg = {
        1: 3340000,
        2: 3645000,
        3: 4050000,
        4: 4635000,
        5: 6530000,
        6: 6530000
    }
    
    cupos_bi = {
        100: 4150000,
        101: 4400000,
        110: 4900000,
        111: 5500000,
        200: 4150000,
        201: 4400000,
        210: 6000000,
        211: 6000000,
        300: 4150000,
        301: 4400000,
        310: 6000000,
        311: 6000000,
        400: 4950000,
        401: 5000000,
        410: 6250000,
        411: 6250000,
        500: 6850000,
        501: 7175000,
        510: 7500000,
        511: 7500000,
        600: 6850000,
        601: 7175000,
        610: 7500000,
        611: 7500000,
    }
    
    cupo = []

    cupo = []

    for i in range(len(df)):
        if df['Nodo Combinado'].iloc[i] in [7, 16]:
            cupo.append(cupos_efg[df['Estrato'].iloc[i]])
        elif (df['Estrato'].iloc[i] in [1, 2, 3, 4, 5, 6]) and (df['Nodo Combinado'].iloc[i] in [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]):
            cupo.append(cupos_bi[df['Nodo codificado'].iloc[i]])
        else:
            cupo.append(0)

    df['Nuevo Cupo'] = cupo

    return df


# COMMAND ----------

storageCS = dbutils.secrets.get(scope='efigas', key='ba-storage-cs')

fnbDF = pd.read_csv(
                        f"abfs://brilla-scoring/rawdata.csv",
                        storage_options={
                                "connection_string":storageCS
                                },
                        encoding='utf8'
                        )

# COMMAND ----------

# Initialize IO variable to host CSV information
output = io.StringIO()
print('Modelo inicializado.')

# COMMAND ----------

fnbDF = preprocess_inputs(fnbDF)
print('Datos procesados.')

# COMMAND ----------

# Run decision tree model
fnbDF = arbol(fnbDF)
print('Arbol terminado.')

# COMMAND ----------

# Assign risk to each contract
fnbDF = riesgo_total(fnbDF)
print('Jerarquizacion de riesgos terminado.')

# COMMAND ----------

# Run decision tree model
fnbDF = nodos(fnbDF)
print('Nodos asignados.')

# COMMAND ----------

# Assign quotas from model output
fnbDF = cupos(fnbDF)
print('Cupos asignados')

# COMMAND ----------

schema = StructType([
    StructField("IdContrato", IntegerType(), True),
    StructField("CupoAsignado", IntegerType(), True),
    StructField("Identificacion", StringType(), True),
    StructField("Tipo", StringType(), True),
    StructField("Nodo", IntegerType(), True),
    StructField("Riesgo", StringType(), True),
    StructField("Categoria", IntegerType(), True),
    StructField("Estrato", IntegerType(), True),
    StructField("FechaPrediccion", DateType(), True)
    ])

today = datetime.now()
today_dt = today.strftime("%Y-%m-%d")

fnbDF = fnbDF[['Contrato','Nuevo Cupo','Identificacion','TipoIdentificacion','Nodo Combinado','Riesgo Combinado', 'Categoria','Estrato']]
fnbDF['FechaPrediccion'] = today_dt
fnbDF['FechaPrediccion'] = pd.to_datetime(fnbDF['FechaPrediccion'])

deltaDF = spark.createDataFrame(fnbDF, schema = schema)

# COMMAND ----------

deltaTableScoring = DeltaTable.forName(spark, 'analiticaefg.brilla.scoringFNB')

deltaTableScoring.alias('scoring') \
  .merge(
    deltaDF.alias('updates'),
    'scoring.IdContrato = updates.IdContrato AND scoring.FechaPrediccion = updates.FechaPrediccion'
  ) \
  .whenMatchedUpdate(set =
    {
      "IdContrato": "updates.IdContrato",
      "CupoAsignado": "updates.CupoAsignado",
      "Identificacion": "updates.Identificacion",
      "Tipo": "updates.Tipo",
      "Nodo": "updates.Nodo",
      "Riesgo": "updates.Riesgo",
      "Categoria": "updates.Categoria",
      "Estrato": "updates.Estrato",
      "FechaPrediccion": "updates.FechaPrediccion"
    }
  ) \
  .whenNotMatchedInsert(values =
    {
      "IdContrato": "updates.IdContrato",
      "CupoAsignado": "updates.CupoAsignado",
      "Identificacion": "updates.Identificacion",
      "Tipo": "updates.Tipo",
      "Nodo": "updates.Nodo",
      "Riesgo": "updates.Riesgo",
      "Categoria": "updates.Categoria",
      "Estrato": "updates.Estrato",
      "FechaPrediccion": "updates.FechaPrediccion"
    }
  ) \
  .execute()
