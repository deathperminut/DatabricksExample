// Databricks notebook source
// DBTITLE 1,Creación de Tabla
// MAGIC %sql
// MAGIC CREATE TABLE IF NOT EXISTS analiticagdc.comercializacion.factvolumen ( 
// MAGIC IdComercializacion  INT,
// MAGIC Fecha               DATE,
// MAGIC Volumen             DECIMAL(20,4)) 
// MAGIC USING DELTA

// COMMAND ----------

// DBTITLE 1,Cálculos
import io.delta.tables._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.expressions.Window

/*Rangos de fechas de interés para consumos*/

val inicio = lit("2021-01-01").cast("date")
val fin = (current_timestamp()-expr("INTERVAL 29 HOURS")).cast("date")

val fechas = DeltaTable.forName("bigdc.comun.dimfecha").toDF
  .filter($"Fecha".between(inicio, fin))
  .select($"Fecha")

/* Tabla con identificadores de historia y columna para los dispositivos, se genera un registro para cada fecha dentro del rango especificado en el paso anterior. */

val dispositivos = DeltaTable.forName("production.comercializacion.estaciones").toDF
  .filter($"codigo_historia".isNotNull and $"columna".isNotNull)
  .select(
    $"id".as("IdComercializacion"),
    $"codigo_historia".as("IdHistoria"),
    $"columna".as("IdColumna"),
    $"signo".as("Signo"),
    $"unidad_medicion".as("Unidad"))
  .crossJoin(fechas)

/* Dataframe final con consumos */

val consumos = dispositivos.as("d")
  .join(
    DeltaTable.forName("bigdc.novo.facthistoriadispositivo").toDF
      .filter($"ValorColumna".isNotNull and
      date_trunc("day", $"FechaHoraHistoria").between(inicio,fin) and
      date_trunc("day", $"FechaHoraHistoria") === $"FechaHoraHistoria" and
      $"is_current" === 1 and
      $"AplicadoDeltaMode" === 1).as("h"),
    $"d.IdHistoria" === $"h.IdHistoriaDispositivo" and
    $"d.IdColumna" === $"h.IdColumna" and
    $"d.Fecha" === date_trunc("day", $"h.FechaHoraHistoria"),
    "left")
  .select(
    $"d.IdComercializacion".cast(IntegerType),
    $"d.Fecha".cast(DateType),
    when($"Unidad" === "M3", $"d.Signo" * $"ValorColumna")
    .when($"Unidad" === "FT3", $"d.Signo" * $"ValorColumna" / 35.3147)
    .when($"Unidad" === "KPC", $"d.Signo" * $"ValorColumna" * 1000 / 35.3147)
    .when($"Unidad" === "MBTU", $"d.Signo" * $"ValorColumna" * 1000 / 35.3147) /* falta poder calorífico */
    .cast(DecimalType(20,4))
    .as("Volumen"),
    row_number.over(Window.partitionBy($"d.IdComercializacion", $"d.Fecha").orderBy($"h.FechaHoraCreacion".desc)).as("rn"))
  .filter($"rn" === 1)
  .drop("rn")

// COMMAND ----------

// DBTITLE 1,Sobreescribir tabla 
consumos.write.mode("overwrite").saveAsTable("analiticagdc.comercializacion.factvolumen")
