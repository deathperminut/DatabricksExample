# Databricks notebook source
# MAGIC %md
# MAGIC ### **Ingesta**

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS analiticagdc.comercializacion.ingesta (
# MAGIC   estacion        VARCHAR(50),
# MAGIC   iddispositivo   VARCHAR(15),
# MAGIC   tipo            VARCHAR(50),
# MAGIC   fecha           DATE,
# MAGIC   volumenm3       FLOAT
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
# MAGIC   estacion          VARCHAR(50),
# MAGIC   fecha             DATE,
# MAGIC   volumenm3         FLOAT,
# MAGIC   festivos          INT,
# MAGIC   diadesemana       INT,
# MAGIC   consumocorregido  FLOAT
# MAGIC )

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

# COMMAND ----------

# MAGIC %sql
# MAGIC TRUNCATE TABLE analiticagdc.comercializacion.estado
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC DELETE FROM analiticagdc.comercializacion.estado
# MAGIC WHERE fecharegistro = '2023-09-10'

# COMMAND ----------


