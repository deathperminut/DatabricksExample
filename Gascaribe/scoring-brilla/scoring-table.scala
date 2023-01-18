// Databricks notebook source
val datalake = "gdcbidatalake.dfs.core.windows.net";

spark.conf.set(s"fs.azure.account.key.$datalake", dbutils.secrets.get(scope="gdcbi", key="datalakekey"));

val goldScoringFNB = s"delta.`abfss://gold@$datalake/scoringFNB`"
spark.sql(s"""
CREATE TABLE IF NOT EXISTS $goldScoringFNB ( 
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
""") 
