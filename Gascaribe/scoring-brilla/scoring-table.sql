-- Databricks notebook source
CREATE TABLE IF NOT EXISTS analiticagdc.brilla.scoringFNB ( 
    IdContrato      BIGINT,
    CupoAsignado    BIGINT,
    Identificacion  VARCHAR(20),
    Tipo            VARCHAR(50),
    Nodo            INTEGER,
    Riesgo          VARCHAR(20),
    Categoria       INTEGER,
    Estrato         INTEGER,
    FechaPrediccion DATE
)

-- COMMAND ----------

CREATE CATALOG analiticagdc

-- COMMAND ----------

USE CATALOG analiticagdc;

-- COMMAND ----------

CREATE DATABASE brilla

-- COMMAND ----------

USE DATABASE brilla;
CREATE TABLE scoringFNB CLONE delta.`abfss://gold@gdcbidatalake.dfs.core.windows.net/scoringFNB`

-- COMMAND ----------

SELECT *
FROM analiticagdc.brilla.scoringFNB
