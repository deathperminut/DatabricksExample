# Databricks notebook source
# MAGIC %md
# MAGIC ### **Ingesta**

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS analiticaefg.comercializacion.ingesta (
# MAGIC   estacion             VARCHAR(50),
# MAGIC   iddispositivo        INT,
# MAGIC   idcomercializacion   INT,
# MAGIC   tipo                 VARCHAR(50),
# MAGIC   fecha                DATE,
# MAGIC   volumenm3            FLOAT
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE analiticaefg.comercializacion.ingesta

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Insumo**

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS analiticaefg.comercializacion.insumo (
# MAGIC   estacion             VARCHAR(50),
# MAGIC   fecha                DATE,
# MAGIC   volumenm3            FLOAT,
# MAGIC   idcomercializacion   INT,
# MAGIC   festivos             INT,
# MAGIC   diadesemana          INT,
# MAGIC   volumen_corregido    FLOAT,
# MAGIC   residuals            FLOAT
# MAGIC
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE analiticaefg.comercializacion.insumo

# COMMAND ----------

# MAGIC %md
# MAGIC ### **RNN**

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS analiticaefg.comercializacion.predicciones_rnn (
# MAGIC   estacion             VARCHAR(50),
# MAGIC   fecha                DATE,
# MAGIC   predicciones         FLOAT,
# MAGIC   error_absoluto       FLOAT,
# MAGIC   modelo               VARCHAR(20),
# MAGIC   estado               VARCHAR(20)
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE analiticaefg.comercializacion.predicciones_rnn
