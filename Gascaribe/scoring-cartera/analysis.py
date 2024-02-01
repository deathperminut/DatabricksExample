# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ### **Se importan librerias**

# COMMAND ----------

import os
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, DateType
from delta.tables import *
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.ml.feature import StandardScaler, VectorAssembler
from datetime import date,datetime
today = datetime.now()
today_dt = today.strftime("%Y-%m-%d")

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ### **Declaración de Variables de Entorno**

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

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ### **Creación de Variables del Modelo**

# COMMAND ----------

df = spark.read.table("analiticagdc.ScoringCartera.FactScoringResidencial") \
    .filter(col("Facturacion") != 0) \
    .withColumn(
        "VarPago",
        when(col("Pagos") > col("Facturacion"), 1)\
        .when(col("Pagos")/col("Facturacion") < 0, 0)\
        .otherwise(col("Pagos")/col("Facturacion"))
    ) \
    .withColumn(
        "VarMora",
        lit(1) - (0.3*col("E30")+ 0.5*col("E60") + 0.8*col("E90") + col("EM90"))/col("Intervalo")
    ) \
    .withColumn(
        "VarRefinanciaciones",
        lit(1) - (col("Refinanciaciones") - min(col("Refinanciaciones")).over(Window.orderBy(lit(1))))/(max(col("Refinanciaciones")).over(Window.orderBy(lit(1))) - min(col("Refinanciaciones")).over(Window.orderBy(lit(1))))
    )
    
    
    #Se remueven productos inactivos y se agrega variable de pago, mora y refinanciaciones

# COMMAND ----------

df6 = df[df['Intervalo'] == 6] \
    .withColumn(
        "DiasSuspendidos",
        when(col("DiasSuspendidos") > 183, 183)
        .otherwise(col("DiasSuspendidos"))
    ) \
    .withColumn(
        "VarSuspensiones",
        when(col("IdTipoProducto") == 7014, lit(1) - (col("DiasSuspendidos") - min(col("DiasSuspendidos")).over(Window.orderBy(lit(1))))/(max(col("DiasSuspendidos")).over(Window.orderBy(lit(1))) - min(col("DiasSuspendidos")).over(Window.orderBy(lit(1))))) \
        .otherwise(None)
    ) \
    .withColumn(
        "VarCastigo",
        col("ConteoCastigado")/max(col("ConteoCastigado")).over(Window.orderBy(lit(1)))
    ) \
    .withColumn(
        "VarDesincronizados",
        when(col("DiasDesincronizados") < 0, lit(1))
        .when(col("DiasDesincronizados") > 60, lit(0))
        .otherwise(lit(1) - col("DiasDesincronizados")/60)
    )

df12 = df[df['Intervalo'] == 12] \
    .withColumn(
        "DiasSuspendidos",
        when(col("DiasSuspendidos") > 366, 366)
        .otherwise(col("DiasSuspendidos"))
    ) \
    .withColumn(
        "VarSuspensiones",
        when(col("IdTipoProducto") == 7014, lit(1) - (col("DiasSuspendidos") - min(col("DiasSuspendidos")).over(Window.orderBy(lit(1))))/(max(col("DiasSuspendidos")).over(Window.orderBy(lit(1))) - min(col("DiasSuspendidos")).over(Window.orderBy(lit(1))))) \
        .otherwise(None)
    ) \
    .withColumn(
        "VarCastigo",
        col("ConteoCastigado")/max(col("ConteoCastigado")).over(Window.orderBy(lit(1)))
    ) \
    .withColumn(
        "VarDesincronizados",
        when(col("DiasDesincronizados") < 0, lit(1))
        .when(col("DiasDesincronizados") > 60, lit(0))
        .otherwise(lit(1) - col("DiasDesincronizados")/60)
    )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ### **Normalización de variables y Análisis de Componentes Principales (PCA)**

# COMMAND ----------

#Transformamos a Pandas para la normalización y para el PCA 
gas6 = df6[df6["IdTipoProducto"] == 7014].toPandas()
gas12 = df12[df12["IdTipoProducto"] == 7014].toPandas()
brilla6 = df6[df6["IdTipoProducto"] != 7014].toPandas()
brilla12 = df12[df12["IdTipoProducto"] != 7014].toPandas()

# COMMAND ----------

gas6["VarPagoNorm"] = StandardScaler().fit_transform(gas6["VarPago"].values.reshape(-1, 1))
gas6["VarMoraNorm"] = StandardScaler().fit_transform(gas6["VarMora"].values.reshape(-1, 1))
gas6["VarRefinanciacionesNorm"] = StandardScaler().fit_transform(gas6["VarRefinanciaciones"].values.reshape(-1, 1))
gas6["VarSuspensionesNorm"] = StandardScaler().fit_transform(gas6["VarSuspensiones"].values.reshape(-1, 1))

# COMMAND ----------

gas12["VarPagoNorm"] = StandardScaler().fit_transform(gas12["VarPago"].values.reshape(-1, 1))
gas12["VarMoraNorm"] = StandardScaler().fit_transform(gas12["VarMora"].values.reshape(-1, 1))
gas12["VarRefinanciacionesNorm"] = StandardScaler().fit_transform(gas12["VarRefinanciaciones"].values.reshape(-1, 1))
gas12["VarSuspensionesNorm"] = StandardScaler().fit_transform(gas12["VarSuspensiones"].values.reshape(-1, 1))

# COMMAND ----------

brilla6["VarPagoNorm"] = StandardScaler().fit_transform(brilla6["VarPago"].values.reshape(-1, 1))
brilla6["VarMoraNorm"] = StandardScaler().fit_transform(brilla6["VarMora"].values.reshape(-1, 1))
brilla6["VarRefinanciacionesNorm"] = StandardScaler().fit_transform(brilla6["VarRefinanciaciones"].values.reshape(-1, 1))
brilla6["VarDesincronizadosNorm"] = StandardScaler().fit_transform(brilla6["VarDesincronizados"].values.reshape(-1, 1))

# COMMAND ----------

brilla12["VarPagoNorm"] = StandardScaler().fit_transform(brilla12["VarPago"].values.reshape(-1, 1))
brilla12["VarMoraNorm"] = StandardScaler().fit_transform(brilla12["VarMora"].values.reshape(-1, 1))
brilla12["VarRefinanciacionesNorm"] = StandardScaler().fit_transform(brilla12["VarRefinanciaciones"].values.reshape(-1, 1))
brilla12["VarDesincronizadosNorm"] = StandardScaler().fit_transform(brilla12["VarDesincronizados"].values.reshape(-1, 1))

# COMMAND ----------

#Realizamos PCA
pca_1 = PCA(n_components=1)

principalComponents_gas6 = pca_1.fit_transform(gas6[["VarPagoNorm", "VarMoraNorm", "VarRefinanciacionesNorm", "VarSuspensionesNorm"]])
principalComponents_gas12 = pca_1.fit_transform(gas12[["VarPagoNorm", "VarMoraNorm", "VarRefinanciacionesNorm", "VarSuspensionesNorm"]])

principalComponents_brilla6 = pca_1.fit_transform(brilla6[["VarPagoNorm", "VarMoraNorm", "VarRefinanciacionesNorm", "VarDesincronizadosNorm"]])
principalComponents_brilla12 = pca_1.fit_transform(brilla12[["VarPagoNorm", "VarMoraNorm", "VarRefinanciacionesNorm", "VarDesincronizadosNorm"]])

# COMMAND ----------

#Escalamos el PCA de 0 a 1 y dejamos en 0 los productos castigados

gas6["Ponderado"] = (principalComponents_gas6 - principalComponents_gas6.max())/(principalComponents_gas6.min() - principalComponents_gas6.max())
gas6["Ponderado"] = (1-gas6["VarCastigo"])*gas6["Ponderado"]

gas12["Ponderado"] = (principalComponents_gas12 - principalComponents_gas12.max())/(principalComponents_gas12.min() - principalComponents_gas12.max())
gas12["Ponderado"] = (1-gas12["VarCastigo"])*gas12["Ponderado"]

gas = pd.concat([gas6, gas12], axis=0)

# COMMAND ----------

#Escalamos el PCA de 0 a 1 y dejamos en 0 los productos castigados

brilla6["Ponderado"] = (principalComponents_brilla6 - principalComponents_brilla6.max())/(principalComponents_brilla6.min() - principalComponents_brilla6.max())
brilla6["Ponderado"] = (1-brilla6["VarCastigo"])*brilla6["Ponderado"]

brilla12["Ponderado"] = (principalComponents_brilla12 - principalComponents_brilla12.max())/(principalComponents_brilla12.min() - principalComponents_brilla12.max())
brilla12["Ponderado"] = (1-brilla12["VarCastigo"])*brilla12["Ponderado"]

brilla = pd.concat([brilla6, brilla12], axis=0)

result = pd.concat([gas, brilla], axis=0)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ### **Algunas visualizaciones y Análisis de Correlación**

# COMMAND ----------

plt.hist(gas["Ponderado"])
plt.show()

# COMMAND ----------

sns.distplot(gas["Ponderado"], hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = gas)

# COMMAND ----------

sns.heatmap(gas[['VarPago','VarMora','VarRefinanciaciones','VarSuspensiones','Castigado','ConteoCastigado','Ponderado']].corr(), annot=True);

# COMMAND ----------

sns.heatmap(brilla[brilla["Intervalo"]==12][['VarPago','VarMora','VarRefinanciaciones','VarDesincronizados','Castigado','ConteoCastigado','Ponderado']].corr(), annot=True);

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ### **Cálculo de los pesos de las Variables por medio de Regresión lineal**

# COMMAND ----------

reg_gas6 = LinearRegression().fit(gas6[['VarPago','VarMora','VarRefinanciaciones','VarSuspensiones']], (principalComponents_gas6 - principalComponents_gas6.max())/(principalComponents_gas6.min() - principalComponents_gas6.max()))

reg_gas12 = LinearRegression().fit(gas12[['VarPago','VarMora','VarRefinanciaciones','VarSuspensiones']], (principalComponents_gas12 - principalComponents_gas12.max())/(principalComponents_gas12.min() - principalComponents_gas12.max()))


reg_brilla6 = LinearRegression().fit(brilla6[['VarPago','VarMora','VarRefinanciaciones','VarDesincronizados']], (principalComponents_brilla6 - principalComponents_brilla6.max())/(principalComponents_brilla6.min() - principalComponents_brilla6.max()))

reg_brilla12 = LinearRegression().fit(brilla12[['VarPago','VarMora','VarRefinanciaciones','VarDesincronizados']], (principalComponents_brilla12 - principalComponents_brilla12.max())/(principalComponents_brilla12.min() - principalComponents_brilla12.max()))
print(reg_gas6.coef_)
print(reg_gas12.coef_)
print(reg_brilla6.coef_)
print(reg_brilla12.coef_)

# COMMAND ----------

#Interceptos de los modelos
print(reg_gas6.intercept_)
print(reg_gas12.intercept_)
print(reg_brilla6.intercept_)
print(reg_brilla12.intercept_)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ### **Segmentación de los Productos por medio del Ponderado**

# COMMAND ----------

def segmentacion(Ponderado):
    if Ponderado <= 0.55 :
        return 0
    elif Ponderado > 0.55 and Ponderado <= 0.7:
        return 1
    elif Ponderado > 0.7 and Ponderado <= 0.85:
        return 2
    elif Ponderado > 0.85 and Ponderado < 1:
        return 3
    elif Ponderado == 1:
        return 4

result["Segmento"] = result["Ponderado"].apply(segmentacion)

result['SegmentoNombre'] = result['Segmento'].replace({
    0:'Pesimo',
    1:'Malo',
    2:'Regular',
    3:'Bueno',
    4:'Excelente'
})

# COMMAND ----------

result[(result["Intervalo"] == 6) & (result["IdTipoProducto"] == 7055)].groupby(["SegmentoNombre"])["SegmentoNombre"].count()

# COMMAND ----------

result[(result["Intervalo"] == 12) & (result["IdTipoProducto"] == 7055)].groupby(["SegmentoNombre"])["SegmentoNombre"].count()

# COMMAND ----------

result['FechaPrediccion'] = today_dt
result['FechaPrediccion'] = pd.to_datetime(result['FechaPrediccion']).dt.strftime('%Y-%m-%d')

result = result.rename(columns={'VarPago':'PagosFacturacion',
                                 'VarMora':'MorasEscaladas',
                                 'VarRefinanciaciones':'RefinanciacionesEscaladas',
                                 'VarSuspensiones':'SuspensionesEscaladas',
                                 'VarDesincronizados':'DesincronizadosEscalados'})

# COMMAND ----------

schema = StructType([
    StructField("IdTipoProducto", IntegerType(), True),
    StructField("IdProducto", IntegerType(), True),
    StructField("Intervalo", IntegerType(), True),  
    StructField("PagosFacturacion", FloatType(), True),
    StructField("MorasEscaladas", FloatType(), True),
    StructField("RefinanciacionesEscaladas", FloatType(), True),
    StructField("SuspensionesEscaladas", FloatType(), True),
    StructField("DiasSuspendidos", IntegerType(), True),
    StructField("DesincronizadosEscalados", FloatType(), True),
    StructField("Castigado", IntegerType(), True),
    StructField("ConteoCastigado", IntegerType(), True),
    StructField("Ponderado", FloatType(), True),
    StructField("Segmento", IntegerType(), True),
    StructField("SegmentoNombre", StringType(), True),
    StructField("FechaPrediccion", DateType(), True)
    ])

df = spark.createDataFrame(result) \
    .select(
        col("IdTipoProducto").cast("int"),
        col("IdProducto").cast("int"),
        col("Intervalo").cast("int"),
        col("PagosFacturacion").cast("float"),
        col("MorasEscaladas").cast("float"),
        col("RefinanciacionesEscaladas").cast("float"),
        col("SuspensionesEscaladas").cast("float"),
        col("DiasSuspendidos").cast("int"),
        col("DesincronizadosEscalados").cast("float"),
        col("Castigado").cast("int"),
        col("ConteoCastigado").cast("int"),
        col("Ponderado").cast("float"),
        col("Segmento").cast("int"),
        col("SegmentoNombre").cast("string"),
        col("FechaPrediccion").cast("date")
    )

# COMMAND ----------

df.write \
.format("com.databricks.spark.sqldw") \
.option("url", sqlDwUrl) \
.option("forwardSparkAzureStorageCredentials", "true") \
.option("dbTable", "ScoringCartera.FactScoring") \
.option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
.mode("overwrite") \
.save()

# COMMAND ----------

df.write.mode('overwrite').saveAsTable('analiticagdc.scoringcartera.factscoring')

# COMMAND ----------

df.write.mode('append').saveAsTable('analiticagdc.scoringcartera.factscoring_historia')
