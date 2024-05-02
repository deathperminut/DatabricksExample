// Databricks notebook source
spark.sql(s"""
CREATE TABLE IF NOT EXISTS analiticagdc.scoringcartera.factscoringresidencial(
    IdTipoProducto          int,
    IdProducto              bigint,
    Intervalo               int,
    E30                     int,
    E60                     int,
    E90                     int,
    EM90                    int,
    Facturacion             bigint,
    Pagos                   bigint,
    Refinanciaciones        int,
    Suspensiones            int,
    DiasSuspendidos         int,
    Castigado               int,
    ConteoCastigado         int,
    DiasDesincronizados     int
)
USING DELTA
""")

// COMMAND ----------

import io.delta.tables._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession
val DimProducto = DeltaTable.forName("bigdc.comun.DimProducto").toDF
val Intervalos = DeltaTable.forName("analiticagdc.ScoringCartera.Intervalos").toDF
val FactResumenCierreDia = DeltaTable.forName("bigdc.Cartera.FactResumenCierreDia").toDF

// COMMAND ----------

val BaseProductos = DimProducto.alias("pr")
  .withColumn(
    "Dia",
    current_date().as("Dia")
  )
  .withColumn(
    "FechaCierre_2",
    last_day(add_months($"Dia",-1))
  )
  .crossJoin(
    Intervalos.alias("i")
  )
  .join(
    FactResumenCierreDia.alias("rc"),
    $"pr.IdProducto" === $"rc.IdProducto" &&
    $"rc.FechaCierre" <= $"FechaCierre_2" &&
    $"rc.FechaCierre" >= last_day(add_months($"FechaCierre_2",-($"i.Intervalo"-1))) &&
    $"pr.FechaIngreso" <= $"FechaCierre_2",
    "inner"
  )
  .filter(
    $"pr.IdCategoria" === 1 &&
    $"pr.IdTipoProducto".isin(7014, 7055) &&
    $"rc.FechaCierre" === last_day($"rc.FechaCierre")
  )
  .withColumn(
    "ConteoCastigado",
    when($"rc.SaldoCastigado" > 0, 1)
    .otherwise(0)
  )
  .select(
    $"rc.FechaCierre",
    $"pr.IdProducto",
    $"pr.IdTipoProducto",
    $"rc.EdadMora",
    $"i.Intervalo",
    $"ConteoCastigado",
    $"rc.IdContrato"
  )

// COMMAND ----------

val BaseCastigado = DimProducto.alias("pr")
  .withColumn(
    "Dia",
    current_date().as("Dia")
  )
  .withColumn(
    "FechaCierre_2",
    last_day(add_months($"Dia",-1))
  )
  .crossJoin(
    Intervalos.alias("i")
  )
  .join(
    FactResumenCierreDia.alias("rc"),
    $"pr.IdProducto" === $"rc.IdProducto" &&
    $"rc.FechaCierre" === last_day($"FechaCierre_2") &&
    $"pr.FechaIngreso" <= $"FechaCierre_2",
    "inner"
  )
  .filter(
    $"pr.IdCategoria" === 1 &&
    $"pr.IdTipoProducto".isin(7014, 7055) &&
    $"rc.FechaCierre" === last_day($"rc.FechaCierre")
  )
  .withColumn(
    "Castigado",
    when($"rc.SaldoCastigado" > 0, 1)
    .otherwise(0)
  )
  .select(
    $"rc.FechaCierre",
    $"pr.IdProducto",
    $"pr.IdTipoProducto",
    $"i.Intervalo",
    $"Castigado"
  )

// COMMAND ----------

val FactFacturacionMensual = DeltaTable.forName("bigdc.Facturacion.FactFacturacionMensual").toDF
val FactPagoMensual = DeltaTable.forName("bigdc.Cartera.FactPagoMensual").toDF

val ComportamientoPago = BaseProductos.alias("bp")
  .join(
    FactFacturacionMensual.alias("ffm"),
    $"bp.IdProducto" === $"ffm.IdProducto" &&
    $"bp.FechaCierre" === $"ffm.FechaCierre",
    "left"
  )
  .join(
    FactPagoMensual.alias("fpm"),
    $"bp.IdProducto" === $"fpm.IdProducto" &&
    $"bp.FechaCierre" === $"fpm.FechaCierre",
    "left"
  )
  .groupBy($"bp.IdProducto", $"bp.IdTipoProducto", $"bp.Intervalo")
  .agg(
    sum(coalesce($"ffm.Valor", lit(0))) as "Facturacion",
    sum(coalesce($"fpm.PagoMes", lit(0))) as "Pagos"
  )

// COMMAND ----------

val DimDiferido = DeltaTable.forName("bigdc.Cartera.DimDiferido").toDF

val Refinanciaciones = BaseProductos.alias("bp")
  .join(
    DimDiferido.alias("dd"),
    $"bp.IdProducto" === $"dd.IdProducto" &&
    $"bp.FechaCierre" === last_day($"dd.FechaIngreso") &&
    $"dd.is_current" === 1 &&
    $"dd.Programa" === "GCNED"
  )
  .withColumn(
    "Refinanciaciones_contar",
    when(isnotnull($"dd.IdProducto"), $"dd.CodigoFinanciacion")
    .otherwise($"dd.IdProducto")
  )
  .groupBy($"bp.IdProducto", $"bp.IdTipoProducto", $"bp.Intervalo")
  .agg(
    countDistinct($"Refinanciaciones_contar") as "Refinanciaciones"
  )

// COMMAND ----------

val BaseSuspensiones = BaseProductos.alias("bp")
  .groupBy($"bp.IdProducto", $"bp.IdTipoProducto", $"bp.Intervalo")
  .agg(
    max($"bp.FechaCierre") as "FechaSuperior",
    min(date_add(last_day(add_months($"bp.FechaCierre", -1)), 1)) as "FechaInferior"
  )

// COMMAND ----------

val FactSuspension = DeltaTable.forName("bigdc.Comun.FactSuspension").toDF

val Suspensiones = BaseSuspensiones.alias("bs")
  .join(
    FactSuspension.alias("fs"),
    $"bs.IdProducto" === $"fs.IdProducto" &&
    $"fs.IdTipoSuspension" === 2 &&
    (
      ($"fs.FechaAplicacion" <= $"bs.FechaInferior" && ($"fs.FechaReconexion" === "1900-01-01" ||
        $"fs.FechaReconexion" > $"bs.FechaInferior")) || //Suspendido antes del periodo
      ($"fs.FechaAplicacion" > $"bs.FechaInferior" && $"fs.FechaAplicacion" <= $"bs.FechaSuperior") //Suspendido durante el periodo
    ),
    "left"
  )
  .withColumn(
    "Suspensiones_contar",
    when(isnotnull($"fs.IdProducto"), $"fs.IdSuspension")
    .otherwise($"fs.IdProducto")
  )
  .withColumn(
    "FechaAplicacion_Inferior",
    when($"fs.FechaAplicacion" < $"bs.FechaInferior", $"bs.FechaInferior")
    .otherwise($"fs.FechaAplicacion")
  )
  .withColumn(
    "FechaReconexion_FechaSuperior",
    when($"fs.FechaReconexion" < $"bs.FechaSuperior", $"fs.FechaReconexion")
    .otherwise($"bs.FechaSuperior")
  )
  .withColumn(
    "DiasSuspendidos_sumar",
    when($"fs.FechaReconexion" === lit("1900-01-01"), datediff($"bs.FechaSuperior", $"FechaAplicacion_Inferior"))
    .otherwise(datediff($"FechaReconexion_FechaSuperior", $"FechaAplicacion_Inferior"))
  )
  .groupBy($"bs.IdProducto", $"bs.IdTipoProducto", $"bs.Intervalo")
  .agg(
    countDistinct($"Suspensiones_contar") as "Suspensiones",
    sum($"DiasSuspendidos_sumar") as "DiasSuspendidos"
  )

// COMMAND ----------

val desincronizados = BaseProductos.alias("bp")
  .withColumn(
    "MORA_GAS",
    when($"IdTipoProducto" === 7014, $"EdadMora")
    .otherwise(null)
  )
  .withColumn(
    "MORA_BRILLA",
    when($"IdTipoProducto" === 7055, $"EdadMora")
    .otherwise(null)
  )
  .groupBy($"FechaCierre", $"IdContrato", $"Intervalo")
  .agg(
    max($"MORA_BRILLA") - max($"MORA_GAS") as "DIF_MORA"
  )
  .groupBy($"IdContrato", $"Intervalo")
  .agg(
    max($"DIF_MORA") as "DiasDesincronizados"
  )


// COMMAND ----------

val ResumenProductos = BaseProductos.alias("bp")
  .withColumn(
    "E30_sumar",
    when($"EdadMora".between(1,30), 1)
    .otherwise(0)
  )
  .withColumn(
    "E60_sumar",
    when($"EdadMora".between(31,60), 1)
    .otherwise(0)
  )
  .withColumn(
    "E90_sumar",
    when($"EdadMora".between(61,90), 1)
    .otherwise(0)
  )
  .withColumn(
    "EM90_sumar",
    when($"EdadMora" > 90, 1)
    .otherwise(0)
  )
  .groupBy($"bp.IdProducto", $"bp.IdTipoProducto", $"bp.Intervalo", $"bp.IdContrato")
  .agg(
    sum($"E30_sumar") as "E30",
    sum($"E60_sumar") as "E60",
    sum($"E90_sumar") as "E90",
    sum($"EM90_sumar") as "EM90",
    sum(coalesce($"bp.ConteoCastigado",lit(0))) as "ConteoCastigado"
  )

// COMMAND ----------

val Scoring = ResumenProductos.alias("bi")
  .join(
    ComportamientoPago.alias("cp"),
    $"bi.IdProducto" === $"cp.IdProducto" &&
    $"bi.Intervalo" === $"cp.Intervalo" &&
    $"bi.IdTipoProducto" === $"cp.IdTipoProducto",
    "left"
  )
  .join(
    Refinanciaciones.alias("ref"),
    $"bi.IdProducto" === $"ref.IdProducto" &&
    $"bi.Intervalo" === $"ref.Intervalo" &&
    $"bi.IdTipoProducto" === $"ref.IdTipoProducto",
    "left"
  )
  .join(
    Suspensiones.alias("susp"),
    $"bi.IdProducto" === $"susp.IdProducto" &&
    $"bi.Intervalo" === $"susp.Intervalo" &&
    $"bi.IdTipoProducto" === $"susp.IdTipoProducto",
    "left"
  )
  .join(
    BaseCastigado.alias("bc"),
    $"bi.IdProducto" === $"bc.IdProducto" &&
    $"bi.Intervalo" === $"bc.Intervalo" &&
    $"bi.IdTipoProducto" === $"bc.IdTipoProducto"
  )
  .join(
    desincronizados.alias("ds"),
    $"bi.IdContrato" === $"ds.IdContrato" &&
    $"bi.Intervalo" === $"ds.Intervalo" && 
    $"bi.IdTipoProducto" === 7055,
    "left"
  )
  .groupBy($"bi.IdTipoProducto", $"bi.IdProducto", $"bi.Intervalo")
  .agg(
    max($"bi.E30").cast("int") as "E30",
    max($"bi.E60").cast("int") as "E60",
    max($"bi.E90").cast("int") as "E90",
    max($"bi.EM90").cast("int") as "EM90",
    sum(coalesce($"cp.Facturacion", lit(0))).cast("bigint") as "Facturacion",
    sum(coalesce($"cp.Pagos", lit(0))).cast("bigint") as "Pagos",
    sum(coalesce($"ref.Refinanciaciones", lit(0))).cast("int") as "Refinanciaciones",
    max(coalesce($"susp.Suspensiones", lit(0))).cast("int") as "Suspensiones",
    max(coalesce($"susp.DiasSuspendidos", lit(0))).cast("int") as "DiasSuspendidos",
    max($"bc.Castigado").cast("int") as "Castigado",
    max($"bi.ConteoCastigado").cast("int") as "ConteoCastigado",
    max(coalesce($"ds.DiasDesincronizados", lit(0))).cast("int") as "DiasDesincronizados"
  )

// COMMAND ----------

Scoring.write.mode("overwrite").saveAsTable("analiticagdc.scoringcartera.factscoringresidencial")
