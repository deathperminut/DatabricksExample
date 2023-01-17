# Databricks notebook source
import pandas as pd
import numpy as np
import os
from azure.storage.blob import *
import io

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

        elif (df['Comport_Pago_Gas'].iloc[i] == '3-No Paga a 90 Dias'):
            node_GDC.append(16)
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

def nodos(df):

    nuevo = []

    for i in range(len(df)):
        if df['Nodo Brilla'].iloc[i] != 'missing':
            nuevo.append(int(df['Nodo Brilla'].iloc[i]))
        elif df['Nodo Brilla'].iloc[i] == 'missing':
            nuevo.append(int(df['Nodo Gas'].iloc[i]))

    df['Nodo Combinado'] = nuevo

    return df

# COMMAND ----------

def cupos(df):

    df = df.copy()

    cupos_gdc = {
        1: 3055000,
        2: 3335000,
        3: 3705000,
        4: 4240000,
        5: 5975000,
        6: 5975000
    }
    
    cupos_bi = {
        1: 4300000,
        2: 4525000,
        3: 5100000,
        4: 5650000,
        5: 6200000,
        6: 6450000,
        7: 6800000
    }
    
    cupo = []

    for i in range(len(df)):
        if df['Nodo Combinado'].iloc[i] in [7, 16]:
            cupo.append(cupos_gdc[df['Estrato'].iloc[i]])
        elif (df['Estrato'].iloc[i] in [1, 2, 3]) and (df['Nodo Combinado'].iloc[i] in [3, 4, 5, 6]):
            cupo.append(cupos_bi[2])
        elif (df['Estrato'].iloc[i] in [1, 2, 3]) and (df['Nodo Combinado'].iloc[i] in [1, 2]):
            cupo.append(cupos_bi[5])
        elif (df['Estrato'].iloc[i] == 4) and (df['Nodo Combinado'].iloc[i] in [3, 4, 5, 6]):
            cupo.append(cupos_bi[4])
        elif (df['Estrato'].iloc[i] == 4) and (df['Nodo Combinado'].iloc[i] in [1, 2]):
            cupo.append(cupos_bi[6])
        elif (df['Estrato'].iloc[i] in [5, 6]) and (df['Nodo Combinado'].iloc[i] in [5, 6]):
            cupo.append(cupos_bi[6])
        elif (df['Estrato'].iloc[i] in [5, 6]) and (df['Nodo Combinado'].iloc[i] in [1, 2, 3, 4]):
            cupo.append(cupos_bi[7])
        elif (df['Estrato'].iloc[i] in [1, 2, 3]) and (df['Nodo Combinado'].iloc[i] in [14, 15]):
            cupo.append(cupos_bi[1])
        elif (df['Estrato'].iloc[i] in [1, 2, 3]) and (df['Nodo Combinado'].iloc[i] in [11, 12, 13]):
            cupo.append(cupos_bi[2])
        elif (df['Estrato'].iloc[i] in [1, 2, 3]) and (df['Nodo Combinado'].iloc[i] in [8, 9, 10]):
            cupo.append(cupos_bi[3])
        elif (df['Estrato'].iloc[i] == 4) and (df['Nodo Combinado'].iloc[i] in [14, 15]):
            cupo.append(cupos_bi[2])
        elif (df['Estrato'].iloc[i] == 4) and (df['Nodo Combinado'].iloc[i] in [11, 12, 13]):
            cupo.append(cupos_bi[3])
        elif (df['Estrato'].iloc[i] == 4) and (df['Nodo Combinado'].iloc[i] in [8, 9, 10]):
            cupo.append(cupos_bi[4])
        elif (df['Estrato'].iloc[i] in [5, 6]) and (df['Nodo Combinado'].iloc[i] in [11, 12, 13, 14, 15]):
            cupo.append(cupos_bi[6])
        elif (df['Estrato'].iloc[i] in [5, 6]) and (df['Nodo Combinado'].iloc[i] in [8, 9, 10]):
            cupo.append(cupos_bi[7])
        else:
            cupo.append('missing')

    df['Nuevo Cupo'] = cupo

    return df


# COMMAND ----------

storageCS = os.environ.get("GDC_BA_STORAGE_CS")

fnbDF = pd.read_csv(
                        f"abfs://brilla-scoring/rawdata.csv",
                        storage_options={
                                "connection_string":storageCS
                                },
                        encoding='utf8'
                        )

# COMMAND ----------

output = io.StringIO()
print('Modelo inicializado.')
fnbDF = preprocess_inputs(fnbDF)
print('Datos procesados.')
fnbDF = arbol(fnbDF)
print('Arbol de Decision terminado.')
fnbDF = riesgo_total(fnbDF)
print('Jerarquizacion de riesgos terminado.')
fnbDF = nodos(fnbDF)
print('Jerarquizacion de nodos terminado.')
fnbDF = cupos(fnbDF)
print('Modelo terminado')
fnbDF = fnbDF[['Contrato','Nuevo Cupo', 'Nodo Combinado','Riesgo Combinado']]
fnbDF.columns = ['contrato','cupo', 'nodo', 'riesgo']
print('Data lista para escribir')

fnbDF.to_csv(output, index=False)

content = output.getvalue()

blob_block = ContainerClient.from_connection_string(
    conn_str=storageCS,
    container_name="brilla-scoring"
)

blob_block.upload_blob('results.csv', content, overwrite=True, encoding='utf-8')

