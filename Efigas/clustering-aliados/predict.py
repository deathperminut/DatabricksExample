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

import sklearn
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

# COMMAND ----------

dwDatabase = dbutils.secrets.get(scope='efigas', key='dwh-name')
dwServer = dbutils.secrets.get(scope='efigas', key='dwh-host')
dwUser = dbutils.secrets.get(scope='efigas', key='dwh-user')
dwPass = dbutils.secrets.get(scope='efigas', key='dwh-pass')
dwJdbcPort = dbutils.secrets.get(scope='efigas', key='dwh-port')
dwJdbcExtraOptions = ""
sqlDwUrl = "jdbc:sqlserver://" + dwServer + ".database.windows.net:" + dwJdbcPort + ";database=" + dwDatabase + ";user=" + dwUser + ";password=" + dwPass + ";" + dwJdbcExtraOptions
storage_account_name = dbutils.secrets.get(scope='efigas', key='bs-name')
blob_container = dbutils.secrets.get(scope='efigas', key='bs-container')
blob_storage = storage_account_name + ".blob.core.windows.net"
config_key = "fs.azure.account.key."+storage_account_name+".blob.core.windows.net"
blob_access_key = dbutils.secrets.get(scope='efigas', key='bs-access-key')
spark.conf.set(config_key, blob_access_key)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### **Data Collection**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC #### **Activos**

# COMMAND ----------

queryCT = 'SELECT * FROM ModeloAliadosBrilla.BaseCT'
queryGS = 'SELECT * FROM ModeloAliadosBrilla.BaseGS'
queryMotos = 'SELECT * FROM ModeloAliadosBrilla.BaseMotos'

# COMMAND ----------

dfCT = spark.read \
  .format("com.databricks.spark.sqldw") \
  .option("url", sqlDwUrl) \
  .option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
  .option("forwardSparkAzureStorageCredentials", "true") \
  .option("maxStrLength", "1024" ) \
  .option("query", queryCT) \
  .load()

rawDataCT = dfCT.toPandas()

rawDataCT = rawDataCT.drop(rawDataCT[rawDataCT['Aliado'] == 'CARDIF COLOMBIA SEGUROS GENERALES S.A.'].index,axis=0)

dfGS = spark.read \
  .format("com.databricks.spark.sqldw") \
  .option("url", sqlDwUrl) \
  .option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
  .option("forwardSparkAzureStorageCredentials", "true") \
  .option("maxStrLength", "1024" ) \
  .option("query", queryGS) \
  .load()

rawDataGS = dfGS.toPandas()

dfMotos = spark.read \
  .format("com.databricks.spark.sqldw") \
  .option("url", sqlDwUrl) \
  .option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
  .option("forwardSparkAzureStorageCredentials", "true") \
  .option("maxStrLength", "1024" ) \
  .option("query", queryMotos) \
  .load()

rawDataMotos = dfMotos.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC #### **Inactivos**

# COMMAND ----------

queryCTInactivo = """SELECT IdContratista
                    FROM Brilla.DimAliado
                    WHERE TipoContratista = 'CANAL TRADICIONAL'
                    AND Valido = 1
                    AND NombreContratista NOT IN ('ALMACEN MOTO CAMPO S.A.S',
                                 'CLASSE MOTOS S.A.S.',
                                 'EJE MOTOS S.A.S',
                                 'GRUPO CREACTIVOS PUBLICIDAD SAS',
                                 'GRUPO SUPER MOTOS S.A.S',
                                 'GRUPO UMA S.A.S',
                                 'IBIZA MOTOS SAS',
                                 'KAWACALDAS',
                                 'LOS OPTIMOTOS S.A.S',
                                 'MAS MOTOS MAS REPUESTOS SAS',
                                 'MOTO PREMIUM  DE OCCIDENTE SAS',
                                 'MOTORED DE COLOMBIA  S.A.S',
                                 'MOTOS DEL RUIZ S.A.S',
                                 'MOTOS LA GRAN PARADA',
                                 'MOTOS Y PARTES DE OCCIDENTE SAS',
                                 'NOCUA MOTOS SAS',
                                 'RIDERS MOTOS SAS',
                                 'RUBE MOTOS SAS',
                                 'SANTA CRUZ DISTRIBUCIONES S.A.S',
                                 'SERVIMOTOS DE OCCIDENTE SAS',
                                 'VERIFICARTE AAA S.A.S',
                                 'YAMAHA DEL CAFÉ S.A.S',
                                 'ZAGAMOTOS DEL PACIFICO S.A.S',
                                 'CARDIF COLOMBIA SEGUROS GENERALES S.A.')
                                 
                   GROUP BY IdContratista"""

queryMotosInactivo = """SELECT IdContratista
                    FROM Brilla.DimAliado
                    WHERE TipoContratista = 'CANAL TRADICIONAL'
                    AND Valido = 1
                    AND NombreContratista IN ('ALMACEN MOTO CAMPO S.A.S',
                                 'CLASSE MOTOS S.A.S.',
                                 'EJE MOTOS S.A.S',
                                 'GRUPO CREACTIVOS PUBLICIDAD SAS',
                                 'GRUPO SUPER MOTOS S.A.S',
                                 'GRUPO UMA S.A.S',
                                 'IBIZA MOTOS SAS',
                                 'KAWACALDAS',
                                 'LOS OPTIMOTOS S.A.S',
                                 'MAS MOTOS MAS REPUESTOS SAS',
                                 'MOTO PREMIUM  DE OCCIDENTE SAS',
                                 'MOTORED DE COLOMBIA  S.A.S',
                                 'MOTOS DEL RUIZ S.A.S',
                                 'MOTOS LA GRAN PARADA',
                                 'MOTOS Y PARTES DE OCCIDENTE SAS',
                                 'NOCUA MOTOS SAS',
                                 'RIDERS MOTOS SAS',
                                 'RUBE MOTOS SAS',
                                 'SANTA CRUZ DISTRIBUCIONES S.A.S',
                                 'SERVIMOTOS DE OCCIDENTE SAS',
                                 'VERIFICARTE AAA S.A.S',
                                 'YAMAHA DEL CAFÉ S.A.S',
                                 'ZAGAMOTOS DEL PACIFICO S.A.S')
                                 
                   GROUP BY IdContratista"""

queryGSInactivo = """SELECT IdPuntoVenta
                    FROM Brilla.DimAliado
                    WHERE TipoContratista = 'GRANDE SUPERFICIE'
                    AND Valido = 1
                    GROUP BY IdPuntoVenta"""

# COMMAND ----------

dfCTInactivo = spark.read \
  .format("com.databricks.spark.sqldw") \
  .option("url", sqlDwUrl) \
  .option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
  .option("forwardSparkAzureStorageCredentials", "true") \
  .option("maxStrLength", "1024" ) \
  .option("query", queryCTInactivo) \
  .load()

rawDataCTInactivo = dfCTInactivo.toPandas()


dfGSInactivo = spark.read \
  .format("com.databricks.spark.sqldw") \
  .option("url", sqlDwUrl) \
  .option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
  .option("forwardSparkAzureStorageCredentials", "true") \
  .option("maxStrLength", "1024" ) \
  .option("query", queryGSInactivo) \
  .load()

rawDataGSInactivo = dfGSInactivo.toPandas()


dfMotosInactivo = spark.read \
  .format("com.databricks.spark.sqldw") \
  .option("url", sqlDwUrl) \
  .option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
  .option("forwardSparkAzureStorageCredentials", "true") \
  .option("maxStrLength", "1024" ) \
  .option("query", queryMotosInactivo) \
  .load()

rawDataMotosInactivo = dfMotosInactivo.toPandas()

# COMMAND ----------

queryDimAliados = "SELECT IdContratista,IdPuntoVenta FROM Brilla.DimAliado WHERE Valido = 1 AND TipoContratista IN ('CANAL TRADICIONAL','GRANDE SUPERFICIE')"

dfAliados = spark.read \
  .format("com.databricks.spark.sqldw") \
  .option("url", sqlDwUrl) \
  .option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
  .option("forwardSparkAzureStorageCredentials", "true") \
  .option("maxStrLength", "1024" ) \
  .option("query", queryDimAliados) \
  .load()

dfAliados = dfAliados.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### **Preprocessing**

# COMMAND ----------

def preprocess_inputs(df):
    df = df.copy()
    df = df[df['Monetario'] > 100_000].copy()
    
    df['Devolucion'] = df['Devoluciones']/df['NumeroDeVentas']

    # Se itera por las columnas Recency y Monetary, y crea bins divididos en quantiles.
    for cols in ['Recencia','Monetario','Devolucion','TicketPromedio','PlazoPromedio','Monetario6M','NumeroDeVentas']:
        # Dependiendo de en cual intervalo (bin) se encuentra, lo clasifica de 5 a 1.
        df[cols] = df[cols].astype('float')
        print(cols)
        if cols == 'Recencia':
            df[f'{cols}_bins'] = pd.cut(df[cols],5)
            sorted_bins = sorted(df[f'{cols}_bins'].value_counts().keys())
            r_score = []
            for j in df[cols]:
                counter = 5
                for v in sorted_bins:
                    if j in v:
                        break
                    else:
                        counter -= 1
                r_score.append(counter)

        elif cols == 'Devolucion':
            r_score = []
            for i in df[cols]:

              if i <= 0.05:
                r_score.append(0)
              else:
                r_score.append(1)

            # Dependiendo de en cual intervalo (bin) se encuentra, lo clasifica de 1 a 5.
        else:
            df[f'{cols}_bins'] = pd.qcut(df[cols],5,duplicates='drop')
            sorted_bins = sorted(df[f'{cols}_bins'].value_counts().keys())
            r_score = []
            for j in df[cols]:
                counter = 0
                for v in sorted_bins:
                    counter += 1
                    if j in v:
                        break
                r_score.append(counter)
                
        df[f'{cols}-Score'] = r_score
   

    df = df.drop(['Recencia_bins','Monetario_bins','NumeroDeVentas_bins','TicketPromedio_bins','Monetario6M_bins','PlazoPromedio_bins'],axis=1)
        
    return df

# COMMAND ----------

XCT = preprocess_inputs(rawDataCT)
XGS = preprocess_inputs(rawDataGS)
XMotos = preprocess_inputs(rawDataMotos)

# COMMAND ----------

def get_na(df,tipoAliado):
    inactivos = df[df['Monetario'] <= 100_000]
    if tipoAliado == 'CT':
        inactivos_df = pd.DataFrame({
                                     'IdContratista':inactivos['IdContratista'],
                                     'cluster':len(inactivos)*[-2],
                                     'name':len(inactivos)*['No Aplica'],
                                     'TipoDeAliado':len(inactivos)*[0], })
        print(f'Numero de Usuarios con Colocacion <= $100,000: {len(inactivos_df)}')
    elif tipoAliado == 'Motos':
        inactivos_df = pd.DataFrame({
                                     'IdContratista':inactivos['IdContratista'],
                                     'cluster':len(inactivos)*[-2],
                                     'name':len(inactivos)*['No Aplica'],
                                     'TipoDeAliado':len(inactivos)*[1], })
        print(f'Numero de Usuarios con Colocacion <= $100,000: {len(inactivos_df)}')
    else:
        inactivos_df = pd.DataFrame({
                                     'IdPuntoVenta':inactivos['IdPuntoVenta'],
                                     'cluster':len(inactivos)*[-2],
                                     'name':len(inactivos)*['No Aplica'],
                                     'TipoDeAliado':len(inactivos)*[2], })
        print(f'Numero de Usuarios con Colocacion <= $100,000: {len(inactivos_df)}')
    return inactivos_df

# COMMAND ----------

XCTNA = get_na(rawDataCT,'CT')
XCTNA = XCTNA.merge(dfAliados, on='IdContratista', how='left')
XGSNA = get_na(rawDataGS,'GS')
XGSNA = XGSNA.merge(dfAliados, on='IdPuntoVenta', how='left')
XMotosNA = get_na(rawDataMotos,'Motos')
XMotosNA = XMotosNA.merge(dfAliados, on='IdContratista', how='left')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### **Canal Tradicional**

# COMMAND ----------

col = ['Recencia-Score',
       'CARDIF',
      'Monetario-Score',
      'Devolucion-Score',
      'TicketPromedio-Score',
      'Monetario6M-Score']


with open(f'/dbfs/ModelosEFG/Aliados/KMeans_CT.pkl', 'rb') as handle:
    model = pkl.load(handle)
    
XCT['cluster'] = model.predict(XCT[col])

centroidsCT = model.cluster_centers_
clustersCT = pd.DataFrame(centroidsCT,columns=col)
clustersCT['cluster'] = model.predict(clustersCT[col]) 
clustersCT['magnitude'] = np.sqrt(((clustersCT['Recencia-Score']**2) + (clustersCT['CARDIF']**2) + (clustersCT['Monetario-Score']**2)+ (clustersCT['Devolucion-Score']**2)+ (clustersCT['TicketPromedio-Score']**2)+ (clustersCT['Monetario6M-Score']**2)))

clustersCT['name'] = [0,0,0,0,0]
clustersCT['name'].iloc[clustersCT['Monetario-Score'].idxmax()] = 'Diamante'
clustersCT['name'].iloc[clustersCT['Monetario-Score'].idxmin()] = 'Bronce'
clustersCT['name'].iloc[clustersCT['Monetario-Score'] == list(clustersCT['Monetario-Score'].nlargest(2))[1]] = 'Platino'
clustersCT['name'].iloc[clustersCT['Monetario-Score'] == list(clustersCT['Monetario-Score'].nsmallest(2))[1]] = 'Plata'
clustersCT['name'].iloc[clustersCT['name'] == 0] = 'Oro'

XCTMerged = XCT.merge(clustersCT[['cluster','name']], on='cluster',how='left')

XCTMerged['TipoDeAliado'] = 0

#XCTMerged = XCTMerged.rename(columns={'IdContratista':'Id'})

XCTMergedAliados = XCTMerged.merge(dfAliados, on='IdContratista', how='left')

# COMMAND ----------

print(f"Numero de datos repetidos: {XCTMergedAliados[['IdContratista','IdPuntoVenta']].duplicated().sum()}")
display(XCTMergedAliados.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Motos**

# COMMAND ----------

col = ['Recencia-Score',
       'CARDIF',
      'Monetario-Score',
      'Devolucion-Score',
      'TicketPromedio-Score',
      'Monetario6M-Score']


with open(f'/dbfs/ModelosEFG/Aliados/KMeans_Motos.pkl', 'rb') as handle:
    model = pkl.load(handle)
    
XMotos['cluster'] = model.predict(XMotos[col])

centroidsMotos = model.cluster_centers_
clustersMotos = pd.DataFrame(centroidsMotos,columns=col)
clustersMotos['cluster'] = model.predict(clustersMotos[col]) 
clustersMotos['magnitude'] = np.sqrt(((clustersMotos['Recencia-Score']**2) + (clustersMotos['CARDIF']**2) + (clustersMotos['Monetario-Score']**2)+ (clustersMotos['Devolucion-Score']**2)+ (clustersMotos['TicketPromedio-Score']**2)+ (clustersMotos['Monetario6M-Score']**2)))

clustersMotos['name'] = [0,0,0,0,0]
clustersMotos['name'].iloc[clustersMotos['Monetario-Score'].idxmax()] = 'Diamante'
clustersMotos['name'].iloc[clustersMotos['Monetario-Score'].idxmin()] = 'Bronce'
clustersMotos['name'].iloc[clustersMotos['Monetario-Score'] == list(clustersMotos['Monetario-Score'].nlargest(2))[1]] = 'Platino'
clustersMotos['name'].iloc[clustersMotos['Monetario-Score'] == list(clustersMotos['Monetario-Score'].nsmallest(2))[1]] = 'Plata'
clustersMotos['name'].iloc[clustersMotos['name'] == 0] = 'Oro'

XMotosMerged = XMotos.merge(clustersMotos[['cluster','name']], on='cluster',how='left')

XMotosMerged['TipoDeAliado'] = 1

#XMotosMerged = XMotosMerged.rename(columns={'IdContratista':'Id'})
XMotosMergedAliados = XMotosMerged.merge(dfAliados, on='IdContratista', how='left')

# COMMAND ----------

print(f"Numero de datos repetidos: {XMotosMergedAliados[['IdContratista','IdPuntoVenta']].duplicated().sum()}")
display(XMotosMerged.head())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ### **Grande Superficie**

# COMMAND ----------

col = ['Recencia-Score',
       'CARDIF',
      'Monetario-Score',
      'Devolucion-Score',
      'TicketPromedio-Score',
      'Monetario6M-Score']


with open(f'/dbfs/ModelosEFG/Aliados/KMeans_GS.pkl', 'rb') as handle:
    model = pkl.load(handle)
    
XGS['cluster'] = model.predict(XGS[col])

centroidsGS = model.cluster_centers_
clustersGS = pd.DataFrame(centroidsGS,columns=col)
clustersGS['cluster'] = model.predict(clustersGS[col]) 
clustersGS['magnitude'] = np.sqrt(((clustersGS['Recencia-Score']**2) + (clustersGS['CARDIF']**2) + (clustersGS['Monetario-Score']**2)+ (clustersGS['Devolucion-Score']**2)+ (clustersGS['TicketPromedio-Score']**2)+ (clustersGS['Monetario6M-Score']**2)))

clustersGS['name'] = [0,0,0,0,0]
clustersGS['name'].iloc[clustersGS['Monetario-Score'].idxmax()] = 'Diamante'
clustersGS['name'].iloc[clustersGS['Monetario-Score'].idxmin()] = 'Bronce'
clustersGS['name'].iloc[clustersGS['Monetario-Score'] == list(clustersGS['Monetario-Score'].nlargest(2))[1]] = 'Platino'
clustersGS['name'].iloc[clustersGS['Monetario-Score'] == list(clustersGS['Monetario-Score'].nsmallest(2))[1]] = 'Plata'
clustersGS['name'].iloc[clustersGS['name'] == 0] = 'Oro'

XGSMerged = XGS.merge(clustersGS[['cluster','name']], on='cluster',how='left')

XGSMerged['TipoDeAliado'] = 2

#XGSMerged = XGSMerged.rename(columns={'IdPuntoVenta':'Id'})

XGSMergedAliados = XGSMerged.merge(dfAliados, on='IdPuntoVenta', how='left')

# COMMAND ----------

print(f"Numero de datos repetidos: {XGSMergedAliados[['IdContratista','IdPuntoVenta']].duplicated().sum()}")
display(XGSMergedAliados.head())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC #### **Procesamiento Inactivos**

# COMMAND ----------

idsActivosCT = XCT['IdContratista'].copy()
idsActivosMotos = XMotos['IdContratista'].copy()
idsActivosGS = XGS['IdPuntoVenta'].copy()
idsNACT = XCTNA['IdContratista'].copy()
idsNAMotos = XMotosNA['IdContratista'].copy()
idsNAGS = XGSNA['IdPuntoVenta'].copy()

idsInactivosCT = list(set(rawDataCTInactivo['IdContratista']) - set(idsActivosCT) - set(idsNACT))
idsInactivosGS = list(set(rawDataGSInactivo['IdPuntoVenta']) - set(idsActivosGS) - set(idsNAGS))
idsInactivosMotos = list(set(rawDataMotosInactivo['IdContratista']) - set(idsActivosMotos) - set(idsNAMotos))

InactivosCT = pd.DataFrame(idsInactivosCT,columns=['IdContratista'])
InactivosCT = InactivosCT.merge(dfAliados, on='IdContratista', how='left')
InactivosGS = pd.DataFrame(idsInactivosGS,columns=['IdPuntoVenta'])
InactivosMotos = pd.DataFrame(idsInactivosMotos,columns=['IdContratista'])
InactivosMotos = InactivosMotos.merge(dfAliados, on='IdContratista', how='left')

# COMMAND ----------

def process_inactivos(df,tipoAliado):
    
    df['cluster'] = -1
    df['name'] = 'Inactivo'
    
    if tipoAliado == 'CT':
        df['TipoDeAliado'] = 0
        #df = df.rename(columns={'IdContratista':'Id'})
    
    elif tipoAliado == 'Motos':
        df['TipoDeAliado'] = 1
        #df = df.rename(columns={'IdContratista':'Id'})
    
    else:
        df['TipoDeAliado'] = 2
        #df = df.rename(columns={'IdPuntoVenta':'Id'})
    
    
    return df

# COMMAND ----------

resultCTInactivo = process_inactivos(InactivosCT,'CT')
resultGSInactivo = process_inactivos(InactivosGS,'GS')
resultGSInactivo = resultGSInactivo.merge(dfAliados, on='IdPuntoVenta', how='left')
resultMotosInactivo = process_inactivos(InactivosMotos,'Motos')

# COMMAND ----------

results = pd.concat([XCTMergedAliados[['IdContratista','IdPuntoVenta','cluster','name','TipoDeAliado']],\
                     XMotosMergedAliados[['IdContratista','IdPuntoVenta','cluster','name','TipoDeAliado']],\
                     XGSMergedAliados[['IdContratista','IdPuntoVenta','cluster','name','TipoDeAliado']],\
                     resultCTInactivo[['IdContratista','IdPuntoVenta','cluster','name','TipoDeAliado']],\
                     resultMotosInactivo[['IdContratista','IdPuntoVenta','cluster','name','TipoDeAliado']],\
                     resultGSInactivo[['IdContratista','IdPuntoVenta','cluster','name','TipoDeAliado']],\
                     XCTNA[['IdContratista','IdPuntoVenta','cluster','name','TipoDeAliado']],\
                     XMotosNA[['IdContratista','IdPuntoVenta','cluster','name','TipoDeAliado']],\
                     XGSNA[['IdContratista','IdPuntoVenta','cluster','name','TipoDeAliado']]],axis=0).reset_index(drop=True)
results = results.rename(columns={'cluster':'Segmento',
                                  'name': 'NombreSegmento'})

results['FechaPrediccion'] = today_dt
results['FechaPrediccion'] = pd.to_datetime(results['FechaPrediccion'])

# COMMAND ----------

print(f"Numero de datos repetidos: {results[['IdContratista','IdPuntoVenta']].duplicated().sum()}")

# COMMAND ----------

results[results['IdContratista'].isna()]

# COMMAND ----------

schema = StructType([
    StructField("IdContratista", IntegerType(), True),
    StructField("IdPuntoVenta", IntegerType(), True),
    StructField("Segmento", IntegerType(), True),
    StructField("NombreSegmento", StringType(), True),
    StructField("TipoDeAliado", IntegerType(), True),
    StructField("FechaPrediccion", DateType(), True)
  ])
df = spark.createDataFrame(results, schema = schema)
df.write \
.format("com.databricks.spark.sqldw") \
.option("url", sqlDwUrl) \
.option("forwardSparkAzureStorageCredentials", "true") \
.option("dbTable", "ModeloAliadosBrilla.StageSegmentosAliados") \
.option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
.mode("overwrite") \
.save()

# COMMAND ----------

groupbyCT = XCTMerged.groupby('name').agg({'Recencia':['count','mean'],'CARDIF':['mean'],'NumeroDeVentas':['mean'],'Monetario':['mean'],'Devolucion':['mean'],'TicketPromedio':['mean'],'Monetario6M':['mean']})
groupbyGS = XGSMerged.groupby('name').agg({'Recencia':['count','mean'],'CARDIF':['mean'],'NumeroDeVentas':['mean'],'Monetario':['mean'],'Devolucion':['mean'],'TicketPromedio':['mean'],'Monetario6M':['mean']})
groupbyMotos = XMotosMerged.groupby('name').agg({'Recencia':['count','mean'],'CARDIF':['mean'],'NumeroDeVentas':['mean'],'Monetario':['mean'],'Devolucion':['mean'],'TicketPromedio':['mean'],'Monetario6M':['mean']})

clusterCT = list(groupbyCT.index)
countCT = groupbyCT['Recencia']['count']
recenciaCT = groupbyCT['Recencia']['mean']
cardifCT = groupbyCT['CARDIF']['mean']
numeroventasCT = groupbyCT['NumeroDeVentas']['mean']
monetarioCT = groupbyCT['Monetario']['mean']
devolucionCT = groupbyCT['Devolucion']['mean']
ticketCT = groupbyCT['TicketPromedio']['mean']
monetario6mCT = groupbyCT['Monetario6M']['mean']

clusterMotos = list(groupbyMotos.index)
countMotos = groupbyMotos['Recencia']['count']
recenciaMotos = groupbyMotos['Recencia']['mean']
cardifMotos = groupbyMotos['CARDIF']['mean']
numeroventasMotos = groupbyMotos['NumeroDeVentas']['mean']
monetarioMotos = groupbyMotos['Monetario']['mean']
devolucionMotos = groupbyMotos['Devolucion']['mean']
ticketMotos = groupbyMotos['TicketPromedio']['mean']
monetario6mMotos = groupbyMotos['Monetario6M']['mean']

clusterGS = list(groupbyGS.index)
countGS = groupbyGS['Recencia']['count']
recenciaGS = groupbyGS['Recencia']['mean']
cardifGS = groupbyGS['CARDIF']['mean']
numeroventasGS = groupbyGS['NumeroDeVentas']['mean']
monetarioGS = groupbyGS['Monetario']['mean']
devolucionGS = groupbyGS['Devolucion']['mean']
ticketGS = groupbyGS['TicketPromedio']['mean']
monetario6mGS = groupbyGS['Monetario6M']['mean']

# COMMAND ----------

dataPandas = pd.DataFrame({'Segmentos': clusterCT,
                           'CantidadDeAliados': countCT,
                           'Recencia': recenciaCT,
                           'CARDIF': cardifCT,
                           'NumeroDeVentas': numeroventasCT,
                           'Monetario': monetarioCT,
                           'Devoluciones': devolucionCT,
                           'TicketPromedio': ticketCT,
                           'Monetario6M': monetario6mCT}).reset_index(drop=True)
schema = StructType([
    StructField("Segmentos", StringType(), True),
    StructField("CantidadDeAliados", IntegerType(), True),
    StructField("Recencia", FloatType(), True),
    StructField("CARDIF", FloatType(), True),
    StructField("NumeroDeVentas", FloatType(), True),
    StructField("Monetario", FloatType(), True),
    StructField("Devoluciones", FloatType(), True),
    StructField("TicketPromedio", FloatType(), True),
    StructField("Monetario6M", FloatType(), True),
    ])
df = spark.createDataFrame(dataPandas, schema = schema)
df.write \
.format("com.databricks.spark.sqldw") \
.option("url", sqlDwUrl) \
.option("forwardSparkAzureStorageCredentials", "true") \
.option("dbTable", "ModeloAliadosBrilla.ResumenCT") \
.option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
.mode("overwrite") \
.save()

# COMMAND ----------

dataPandas = pd.DataFrame({'Segmentos': clusterMotos,
                           'CantidadDeAliados': countMotos,
                           'Recencia': recenciaMotos,
                           'CARDIF': cardifMotos,
                           'NumeroDeVentas': numeroventasMotos,
                           'Monetario': monetarioMotos,
                           'Devoluciones': devolucionMotos,
                           'TicketPromedio': ticketMotos,
                           'Monetario6M': monetario6mMotos}).reset_index(drop=True)
schema = StructType([
    StructField("Segmentos", StringType(), True),
    StructField("CantidadDeAliados", IntegerType(), True),
    StructField("Recencia", FloatType(), True),
    StructField("CARDIF", FloatType(), True),
    StructField("NumeroDeVentas", FloatType(), True),
    StructField("Monetario", FloatType(), True),
    StructField("Devoluciones", FloatType(), True),
    StructField("TicketPromedio", FloatType(), True),
    StructField("Monetario6M", FloatType(), True),
    ])
df = spark.createDataFrame(dataPandas, schema = schema)
df.write \
.format("com.databricks.spark.sqldw") \
.option("url", sqlDwUrl) \
.option("forwardSparkAzureStorageCredentials", "true") \
.option("dbTable", "ModeloAliadosBrilla.ResumenMotos") \
.option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
.mode("overwrite") \
.save()

# COMMAND ----------

dataPandas = pd.DataFrame({'Segmentos': clusterGS,
                           'CantidadDeAliados': countGS,
                           'Recencia': recenciaGS,
                           'CARDIF': cardifGS,
                           'NumeroDeVentas': numeroventasGS,
                           'Monetario': monetarioGS,
                           'Devoluciones': devolucionGS,
                           'TicketPromedio': ticketGS,
                           'Monetario6M': monetario6mGS}).reset_index(drop=True)
schema = StructType([
    StructField("Segmentos", StringType(), True),
    StructField("CantidadDeAliados", IntegerType(), True),
    StructField("Recencia", FloatType(), True),
    StructField("CARDIF", FloatType(), True),
    StructField("NumeroDeVentas", FloatType(), True),
    StructField("Monetario", FloatType(), True),
    StructField("Devoluciones", FloatType(), True),
    StructField("TicketPromedio", FloatType(), True),
    StructField("Monetario6M", FloatType(), True),
    ])
df = spark.createDataFrame(dataPandas, schema = schema)
df.write \
.format("com.databricks.spark.sqldw") \
.option("url", sqlDwUrl) \
.option("forwardSparkAzureStorageCredentials", "true") \
.option("dbTable", "ModeloAliadosBrilla.ResumenGS") \
.option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
.mode("overwrite") \
.save()

# COMMAND ----------

print()

# COMMAND ----------


