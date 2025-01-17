# Databricks notebook source
import os
import pandas as pd
import numpy as np
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import *     
from delta.tables import *
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import itertools
import math
import random
import time
from datetime import date,datetime,timedelta,timezone
from pandas.api.indexers import BaseIndexer

import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Funciones

# COMMAND ----------

def completar_fechas(df):
    
    hoy =  date_sub(to_date( from_utc_timestamp(current_timestamp(), 'GMT-5') ), 1)
    
    estaciones = df.groupBy("IdComercializacion", "Estacion", "TipoUsuario").agg( min("Fecha").alias("FechaMimima") )

    fechas_estaciones = estaciones.alias("e") \
              .join( DeltaTable.forName(spark, 'bigdc.comun.dimfecha').toDF().alias("f") , (col("f.Fecha") >= col("e.FechaMimima")) & (col("f.Fecha") <= lit(hoy) ), 'left' ) \
              .withColumn( 'Festivo', when( col("TipoDia") == lit("FESTIVOS"), lit(1) ).otherwise(lit(0)) ) \
              .withColumn( 'DiaDeS', when( col("DiaSemana") == lit("Lunes"), lit(0) )
                                    .when( col("DiaSemana") == lit("Martes"), lit(1) )
                                    .when( col("DiaSemana") == lit("Miércoles"), lit(2) )
                                    .when( col("DiaSemana") == lit("Jueves"), lit(3) )
                                    .when( col("DiaSemana") == lit("Viernes"), lit(4) )
                                    .when( col("DiaSemana") == lit("Sábado"), lit(5) )
                                    .when( col("DiaSemana") == lit("Domingo"), lit(6) ) )  \
              .selectExpr(
                  "Fecha",
                  "DiaDeS as DiaSemana",
                  "Festivo",
                  "IdComercializacion",
                  "Estacion",
                  "TipoUsuario"
              )

    ingesta_completa = fechas_estaciones.alias("fe").join(df.alias("df"), 
                                                      (col("fe.IdComercializacion") == col("df.IdComercializacion")) & 
                                                      (col("fe.Fecha") == col("df.Fecha")), 'left'  ) \
                                                .withColumn( 'Volumen_', when(col("df.Volumen").isNull(), lit(0)).otherwise(col("df.Volumen"))  )  \
                                                .selectExpr(
                                                    "fe.*",
                                                    "Volumen_ as Volumen"
                                                )
                        
    return ingesta_completa

# COMMAND ----------

def criterio_estado(df, ventana=30, new_until=60):
    
    hoy =  date_sub(to_date( from_utc_timestamp(current_timestamp(), 'GMT-5') ), 1)
    hace_n_dias =  date_sub(hoy, ventana-1)
    
    estaciones_fechas = df

    estado_estaciones_ = estaciones_fechas.alias("v") \
     .groupBy("IdComercializacion", "Estacion", "TipoUsuario", "PrimeraFechaEfectiva") \
     .agg(
      min( col("v.Fecha") ).alias("PrimeraFecha"),
      min( when( (estaciones_fechas.Volumen != 0) & (estaciones_fechas.Fecha >= lit(hace_n_dias)), estaciones_fechas.Fecha ).otherwise( lit( date_add(hoy, 1) ) ) ).alias("PrimeraFechaConTrasmision"),
      sum( when( (col("v.Volumen") == 0) & (estaciones_fechas.Fecha >= lit(hace_n_dias) ), lit(1)  ).otherwise(lit(0)) ).alias("DiasSinTrasmisionV") ) \
    .withColumn( 'DiasSeguidosSinTrasmision', date_diff( "PrimeraFechaConTrasmision", lit(hace_n_dias) ) ) \
    .withColumn( 'DiasCompletacionVentana2', lit(ventana) - col("DiasSeguidosSinTrasmision") ) \
    .withColumn( 'PorcentajeConsumoV', (lit(ventana) - col("DiasSinTrasmisionV"))/lit(ventana) ) \
    .withColumn( 'Hoy',  lit(hoy) ) \
    .withColumn( 'Hace_n', lit(hace_n_dias) ) \
    .withColumn( 'FechaInicialVentana2', when( (col("DiasCompletacionVentana2") > 0) & (col("DiasCompletacionVentana2") < 
                lit(ventana)), date_sub( lit(hace_n_dias), col("DiasCompletacionVentana2") ) ).otherwise(lit(None)) ) \
    .withColumn( 'DST_estaciones_recien_aparecidas', when( col("PrimeraFecha") > col("FechaInicialVentana2"), lit 
                     (ventana) ).otherwise(lit(None)) )
    

    estado_estaciones_pot_nuevas = estaciones_fechas.alias("v") \
        .join( estado_estaciones_.alias("ee"), (col("ee.IdComercializacion") == col("v.IdComercializacion")) & (col("ee.FechaInicialVentana2").isNotNull()), 'inner' ) \
        .groupBy("v.IdComercializacion", "v.Estacion") \
        .agg( sum( when( (col("v.Volumen") == 0) & (estaciones_fechas.Fecha < col("ee.PrimeraFechaConTrasmision")) & (estaciones_fechas.Fecha >= col("FechaInicialVentana2") ), lit(1)  ).otherwise(lit(0)) ).alias("DiasSinTrasmisionV2_") ) 
        

    estado_estaciones =  estado_estaciones_.alias("ee") \
        .join( estado_estaciones_pot_nuevas.alias("een"), (col("een.IdComercializacion") == col("ee.IdComercializacion")), 'left' ) \
        .withColumn( 'DiasSinTrasmisionV2', when(col("ee.DST_estaciones_recien_aparecidas").isNull(),col("een.DiasSinTrasmisionV2_") ).otherwise(col("DST_estaciones_recien_aparecidas")) ) \
        .selectExpr( "ee.*", "DiasSinTrasmisionV2" ) \
        .withColumn( 'Estado', when( (col("DiasSinTrasmisionV2").isNull()) & (col("DiasSeguidosSinTrasmision") >= lit 
                         (ventana)) , lit("NO ACTIVA") )
                         .when( (col("DiasSinTrasmisionV2").isNull()) & (col("DiasSeguidosSinTrasmision") < lit(ventana))
                               & ( date_diff( col("Hoy"), col("PrimeraFechaEfectiva") ) > lit(new_until) ), lit("ACTIVA") )
                         .when( ( col("DiasSinTrasmisionV2") < lit(ventana)  ) & ( date_diff( col("Hoy"), col("PrimeraFechaEfectiva") ) > lit(new_until) ), lit("ACTIVA") ) 
                         .when( ( col("DiasSinTrasmisionV2") == lit(ventana) ) | ( date_diff( col("Hoy"), col("PrimeraFechaEfectiva") ) <= lit(new_until) ) , lit("NUEVA") ) )
    
                            
    return estado_estaciones

# COMMAND ----------

def primera_fecha_efectiva(df, n=30):
   
    hoy = date_sub(to_date( from_utc_timestamp(current_timestamp(), 'GMT-5') ), 1)
    
    # Creación de ventanas
    # Ventana para calcular las ventanas de inactividad
    window1 = Window.partitionBy("IdComercializacion").orderBy("Fecha").rowsBetween(-n, -1)
    # Ventana para filtrar la última fecha de la ultima ventana de inactividad
    window2 = Window.partitionBy("IdComercializacion").orderBy("Diferencia")

    # columna que contenga la suma acumulativa de valores cero dentro de la ventana
    df_ = df.withColumn("zero_sum", sum(col("Volumen")).over(window1))
    
    # Fecha minima global, para complementar estaciones que no tengas oasis de inactividad
    fecha_minima = df.groupBy( "IdComercializacion", "Estacion", "TipoUsuario" ) \
    .agg( min(col("Fecha")).alias("FechaMinima") )

    # Encontrar el último valor de la última ventana de tamaño n con valores cero
    ultima_ventana = df_.filter( (col("zero_sum") == 0) & (col("Volumen") != 0) ) \
    .withColumn( 'Diferencia', date_diff(lit(hoy), col("Fecha")) ) \
    .withColumn( 'RN', row_number().over(window2) ) \
    .filter(  col("RN") == 1  )
    
    # Calculo de la primera fecha efectiva comparando el ultima valor de la ultima ventana con la primera fecha
    primeras_fechas_efectivas = fecha_minima.alias("fm") \
    .join( ultima_ventana.alias("uv"), col("fm.IdComercializacion") == col("uv.IdComercializacion") , "left" ) \
    .withColumn( 'PrimeraFechaEfectiva', when( col("uv.Fecha").isNull(), col("fm.FechaMinima") )
                                        .otherwise(col("uv.Fecha")) ) \
    .selectExpr( "fm.IdComercializacion", "fm.Estacion", "fm.TipoUsuario", "PrimeraFechaEfectiva" )

    # Union de la primera fecha en el dataframe de insumo
    insumo = df.alias("d") \
        .join( primeras_fechas_efectivas.alias("pfe"), col("d.IdComercializacion") == col("pfe.IdComercializacion") , "left" ) \
        .selectExpr( "d.*", "pfe.PrimeraFechaEfectiva" )    

    

    return insumo

# COMMAND ----------

from scipy.stats import chi2

def metricas_relevantes(ventana=15):

    schema = StructType([
        StructField("IdComercializacion", IntegerType(), True),
        StructField("PrimeraFechaEfectiva", DateType(), True),
        StructField("Porcentaje", FloatType(), True),
        StructField("NumeroCeros", IntegerType(), True),
        StructField("NumeroTotal", IntegerType(), True),
        StructField("VolumenPromedio", FloatType(), True),
        StructField("VolumenDesviacion", FloatType(), True),
        StructField("Uniformidad", FloatType(), True),
        ])


    @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
    def metricas_relevantes__(df):
        df = df.copy()

        df['Flag_0'] = np.where( df['Volumen'] == 0, 1, 0 )
                
        result_ = df.groupby( ['IdComercializacion', 'PrimeraFechaEfectiva'] ).agg( {'Flag_0':['sum', 'count'], 'Volumen':['mean', 'std']} ).reset_index()
        result_['Porcentaje'] = 1 - ((result_['Flag_0']['sum'])/(result_['Flag_0']['count']))
        result_['NumeroCeros'] = result_['Flag_0']['sum']
        result_['NumeroTotal'] = result_['Flag_0']['count']
        result_['VolumenPromedio'] = result_['Volumen']['mean']
        result_['VolumenDesviacion'] = result_['Volumen']['std']

        bondad_test_ = df.reset_index()
        bondad_test_['NumeroVentana'] = bondad_test_['index']//ventana

        bondad_test = bondad_test_.groupby( ['IdComercializacion', 'NumeroVentana'] ).agg( 'sum' ).reset_index()
        bondad_test = bondad_test[['IdComercializacion', 'NumeroVentana', 'Flag_0']]
        bondad_test['Esperado'] =  (result_['NumeroCeros'].iloc[0]/result_['NumeroTotal'].iloc[0])*ventana
        bondad_test['Estadistico_i'] = ( (bondad_test['Esperado'] - bondad_test['Flag_0'])*(bondad_test['Esperado'] - bondad_test['Flag_0']) )/bondad_test['Esperado']
   
        result_['Uniformidad'] =  (1 - chi2.cdf(np.sum(bondad_test['Estadistico_i']), len(bondad_test['Esperado'])-1 )) 


        result = result_[['IdComercializacion', 'PrimeraFechaEfectiva', 'Porcentaje', 'NumeroCeros', 'NumeroTotal', 'VolumenPromedio', 'VolumenDesviacion', 'Uniformidad']]
    
        return result 
    
    return metricas_relevantes__

# COMMAND ----------

def prophet_filter(n1=30,n2=15):

    schema = StructType([
        StructField("Fecha", DateType(), True),
        StructField("DiaSemana", IntegerType(), True),
        StructField("Festivo", IntegerType(), True),
        StructField("IdComercializacion", IntegerType(), True),
        StructField("Estacion", StringType(), True),
        StructField("TipoUsuario", StringType(), True),
        StructField("Volumen", FloatType(), True),
        StructField("VolumenCorregido", FloatType(), True),
        StructField("deviation", FloatType(), True),
        StructField("standar_deviation1", FloatType(), True),
        StructField("standar_deviation2", FloatType(), True),
        StructField("Estado", StringType(), True),
        StructField("PrimeraFechaEfectiva", DateType(), True),
        ])


    @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
    def prophet_filter__(df):
        df = df.copy()
        df['ds'] = df['Fecha']
        df['y'] = df['Volumen']

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

        def sd_my(x):
            return np.sqrt( x.sum()/(len(x) - 1) )


        
        holidays = df[df['Festivo'] == 1].reset_index(drop=True).sort_values(by='Fecha')[['Fecha', 'Volumen']]
        holidays['holiday'] = 'Festivo_'
        holidays = holidays[['holiday', 'Fecha']]
        holidays.columns = ['holiday', 'ds']
        holidays['ds'] = pd.to_datetime(holidays['ds'])
        _ = df
        _['ds'] = pd.to_datetime(_['ds'])
        model = Prophet(changepoint_prior_scale = 0.5 , holidays = holidays)
        model.fit(_[['ds','y']])

        future = model.make_future_dataframe(periods=0)
        forecast = model.predict(future)
        temporary_df = _.merge(forecast,how='left',on='ds')

        temporary_df['deviationsq'] = (temporary_df['y'] - temporary_df['trend'])*(temporary_df['y'] - temporary_df['trend'])
        temporary_df['deviation'] = (temporary_df['y'] - temporary_df['trend'])
        indexer = CustomIndexer(window_size=n1)
        indexer2 = CustomIndexer(window_size=n2)
        temporary_df['standar_deviation1'] = temporary_df['deviationsq'].rolling(indexer,  min_periods=1).apply(sd_my)
        temporary_df['standar_deviation2'] = temporary_df['deviationsq'].rolling(indexer2,  min_periods=1).apply(sd_my)
        temporary_df['outlier_flag'] = np.where( ( ( temporary_df['y'] > temporary_df['yhat_upper'] ) | ( temporary_df['y'] < temporary_df['yhat_lower'] ) ) & ( (1.5*temporary_df['standar_deviation1']) < (np.abs(temporary_df['deviation']))  ) & ( (1.5*temporary_df['standar_deviation2']) < (np.abs(temporary_df['deviation'])) ), True , False )

        temporary_df['VolumenCorregido'] = None

        for i in range(len(temporary_df)):
            if temporary_df['outlier_flag'][i]:
                if temporary_df['trend'][i] >= 0:
                    temporary_df['VolumenCorregido'][i] = temporary_df['trend'][i] 
                else: 
                    temporary_df['VolumenCorregido'][i] = 0

            else:
                temporary_df['VolumenCorregido'][i] = temporary_df['y'][i]

        result = temporary_df[['Fecha', 'DiaSemana', 'Festivo', 'IdComercializacion', 'Estacion', 'TipoUsuario', 'Volumen', 'VolumenCorregido', 'deviation', 'standar_deviation1', 'standar_deviation2', 'Estado', 'PrimeraFechaEfectiva']]
    
        return result 
    
    return prophet_filter__

# COMMAND ----------

# MAGIC %md
# MAGIC ## Procedimientos Insumo y DimEstado

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS analiticagdc.comercializacion.insumo (
# MAGIC   Fecha                   date,
# MAGIC   DiaSemana               int,
# MAGIC   Festivo                 int,
# MAGIC   IdComercializacion      int,
# MAGIC   Estacion                varchar(100),   
# MAGIC   TipoUsuario             varchar(100),
# MAGIC   Volumen                 float,
# MAGIC   VolumenCorregido        float,
# MAGIC   deviation               float,
# MAGIC   standar_deviation1      float,
# MAGIC   standar_deviation2      float,
# MAGIC   Estado                  varchar(20),
# MAGIC   PrimeraFechaEfectiva    date
# MAGIC )

# COMMAND ----------

ingesta = DeltaTable.forName(spark, 'analiticagdc.comercializacion.ingesta').toDF()
dimestado = DeltaTable.forName(spark, 'analiticagdc.comercializacion.dimestado').toDF()

# COMMAND ----------

estaciones_fechas = completar_fechas(ingesta)
estaciones_primera_fecha = primera_fecha_efectiva(estaciones_fechas, n=30)
estado_estaciones = criterio_estado(estaciones_primera_fecha, ventana=30, new_until=60)

metricas_15 = metricas_relevantes(ventana=15)
metricas = estaciones_primera_fecha.groupBy( col("IdComercializacion")).apply(metricas_15)

estaciones_filtro_primera_fecha = metricas.filter( ((col("Porcentaje") < 0.85) & (col("Uniformidad") < 0.05)) ).selectExpr( 'IdComercializacion' )

insumo_sin_filtro_correcion_primerafecha = estaciones_primera_fecha.filter( (col("Fecha") >= col("PrimeraFechaEfectiva")) & (col("IdComercializacion").isin( estaciones_filtro_primera_fecha.rdd.flatMap(lambda x: x).collect() ) ) )
insumo_sin_filtro_no_correcion = estaciones_primera_fecha.filter( ~(col("IdComercializacion").isin( estaciones_filtro_primera_fecha.rdd.flatMap(lambda x: x).collect() ) ) )

insumo_sin_filtro_ = insumo_sin_filtro_correcion_primerafecha.union(insumo_sin_filtro_no_correcion).alias("ef") \
    .join( estado_estaciones.alias("ee"), col("ee.IdComercializacion") == col("ef.IdComercializacion"), 'left' ) \
    .selectExpr( "ef.*", "ee.Estado" )


estaciones_1registro = insumo_sin_filtro_.groupBy(col("IdComercializacion")).agg( count(col("IdComercializacion")).alias("Cuenta") ).filter( col("Cuenta") == 1 ).selectExpr( 'IdComercializacion' )


insumo_sin_filtro = insumo_sin_filtro_.filter( ~( col("IdComercializacion").isin( estaciones_1registro.rdd.flatMap(lambda x: x).collect() ) ) ) 
insumo_estacion_1registro = insumo_sin_filtro_.filter( ( col("IdComercializacion").isin( estaciones_1registro.rdd.flatMap(lambda x: x).collect() ) ) ) 

# COMMAND ----------

insumo_sin_filtro2 = insumo_sin_filtro
# Partition the data
insumo_sin_filtro2.createOrReplaceTempView("insumo_sf")
sql = "select * from insumo_sf"
insumo_sin_filtro_partition = (spark.sql(sql)\
   .repartition(spark.sparkContext.defaultParallelism, 
   ['IdComercializacion'])).cache()
insumo_sin_filtro_partition.explain()

# COMMAND ----------

prophet_filter_30_15 = prophet_filter(n1=30,n2=15)
insumo_filtro_ = insumo_sin_filtro_partition.groupby(['IdComercializacion']).apply(prophet_filter_30_15)

insumo_filtro_1registro = insumo_estacion_1registro.withColumn('VolumenCorregido', col("Volumen") ) \
                .withColumn('deviation', lit(0)) \
                .withColumn('standar_deviation1', lit(0)) \
                .withColumn('standar_deviation2', lit(0)) \
                .selectExpr( 'Fecha', 'DiaSemana', 'Festivo', 'IdComercializacion', 'Estacion', 'TipoUsuario', 'Volumen', 'VolumenCorregido', 'deviation', 'standar_deviation1', 'standar_deviation2', 'Estado', 'PrimeraFechaEfectiva' )

insumo_filtro = insumo_filtro_.union(insumo_filtro_1registro)

# COMMAND ----------

estados = estado_estaciones.alias("ee") \
.join( dimestado.alias("de"), (col("ee.IdComercializacion") == col("de.IdComercializacion")) & (col("de.is_current") == 1), 'left' ) \
.withColumn( 'Operacion', when( (col("de.Estado").isNull()), lit("INSERTAR") )
                         .when( (col("ee.Estado") != col("de.Estado")), lit("ACTUALIZAR") )  ) \
.withColumn( 'is_current', lit(True) ) \
.selectExpr( "ee.*", "Operacion", "de.Estado as ViejoEstado", "de.FechaRegistro", "is_current" )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merge tabla Insumo

# COMMAND ----------

insumo_filtro.write.mode("overwrite").saveAsTable("analiticagdc.comercializacion.insumo")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merge Tabla DimEstado

# COMMAND ----------

deltaTable_dimestado = DeltaTable.forName(spark, 'analiticagdc.comercializacion.dimestado')

insertar =  {
              "IdComercializacion"  : "df.IdComercializacion",
              "Estacion"            : "df.Estacion",
              "Estado"              : "df.Estado",
              "FechaRegistro"       : from_utc_timestamp(current_timestamp(), 'GMT-5'),
              "is_current"          : lit(True)
    }

actualizar_0 =  {
              "IdComercializacion"  : "df.IdComercializacion",
              "Estacion"            : "df.Estacion",
              "Estado"              : "df.Estado",
              "FechaRegistro"       : "df.FechaRegistro",
              "is_current"          : lit(False)
    }


deltaTable_dimestado.alias('t') \
  .merge( estados.filter("Operacion = 'ACTUALIZAR'").alias('df'), 't.IdComercializacion = df.IdComercializacion AND t.is_current = True') \
  .whenMatchedUpdate(set=actualizar_0) \
  .execute()

deltaTable_dimestado.alias('t') \
  .merge( estados.filter("Operacion = 'ACTUALIZAR'").alias('df'), 't.IdComercializacion = df.IdComercializacion AND t.is_current = df.is_current') \
  .whenNotMatchedInsert(values=insertar) \
  .execute()

deltaTable_dimestado.alias('t') \
  .merge( estados.filter("Operacion = 'INSERTAR'").alias('df'), 'False') \
  .whenNotMatchedInsert(values=insertar) \
  .execute()
