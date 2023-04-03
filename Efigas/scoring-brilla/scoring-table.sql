-- Databricks notebook source
CREATE TABLE IF NOT EXISTS analiticaefg.brilla.scoringFNB ( 
    IdContrato      BIGINT,
    CupoAsignado    BIGINT,
    Identificacion  VARCHAR(100),
    Tipo            VARCHAR(100),
    Nodo            INTEGER,
    Riesgo          VARCHAR(20),
    Categoria       INTEGER,
    Estrato         INTEGER,
    FechaPrediccion DATE
)
