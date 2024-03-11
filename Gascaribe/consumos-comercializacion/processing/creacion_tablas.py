# Databricks notebook source
# MAGIC %md
# MAGIC ### **Ingesta**

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS analiticagdc.comercializacion.ingesta (
# MAGIC   IdComercializacion int,
# MAGIC   Estacion           varchar(50),
# MAGIC   IdDispositivo      int,
# MAGIC   --Tipo               varchar(50),
# MAGIC   Fecha              date,
# MAGIC   Volumen            float
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC select *
# MAGIC from analiticagdc.comercializacion.ingesta
# MAGIC order by estacion,fecha

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Insumo**

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
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC select *
# MAGIC from analiticagdc.comercializacion.insumo
# MAGIC order by estacion,fecha

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Estado de Estaciones**

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS analiticagdc.comercializacion.dimestado (
# MAGIC   IdComercializacion int,
# MAGIC   Estacion           varchar(100),
# MAGIC   Estado             varchar(20),
# MAGIC   FechaRegistro      timestamp,
# MAGIC   is_current         boolean
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS analiticagdc.comercializacion.estado (
# MAGIC   estacion          VARCHAR(50),
# MAGIC   estado            VARCHAR(10),
# MAGIC   fecharegistro     DATE
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC select *
# MAGIC from analiticagdc.comercializacion.estado
# MAGIC order by estacion,fecharegistro
