// Databricks notebook source
// MAGIC %sql
// MAGIC CREATE TABLE IF NOT EXISTS analiticagdc.brilla.insumosscoring(
// MAGIC   TipoIdentificacion     varchar(100),
// MAGIC   Identificacion         varchar(30),
// MAGIC   IdContrato             bigint,
// MAGIC   Comport_Pago_Gas       varchar(25),
// MAGIC   Comport_Pago_Brilla    varchar(25),
// MAGIC   Edad0Gas               int,
// MAGIC   Edad30Gas              int,
// MAGIC   Edad60Gas              int,
// MAGIC   Edad90Gas              int,
// MAGIC   EdadM90Gas             int,
// MAGIC   Edad0Brilla            int,
// MAGIC   Edad30Brilla           int,
// MAGIC   Edad60Brilla           int,
// MAGIC   Edad90Brilla           int,
// MAGIC   EdadM90Brilla          int,
// MAGIC   Tipo_Cupo              varchar(13) not null,
// MAGIC   Departamento           varchar(40),
// MAGIC   IdCategoria            smallint,
// MAGIC   IdSubCategoria         smallint,
// MAGIC   CuotasPendientesGas    int,
// MAGIC   CuotasPendientesBrilla int,
// MAGIC   CuotaGas               int,
// MAGIC   CuotaBrilla            int
// MAGIC )

// COMMAND ----------

import io.delta.tables._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession

val localidadesdrop = DeltaTable.forName("bigdc.comun.dimgeografia").toDF.filter($"IdGeografia".isin(143, 157, 182, 32, 30)).select("IdGeografia")

val FactResumenCierreDia = DeltaTable.forName("bigdc.Cartera.FactResumenCierreDia").toDF.alias("rc")
  .join(
    localidadesdrop.alias("ld"),
    $"rc.IdBarrio" === $"ld.IdGeografia",
    "left"
  )
  .filter(
    $"IdGeografia".isNull
  )
val DimDiferido = DeltaTable.forName("bigdc.Cartera.DimDiferido").toDF
val DimProducto = DeltaTable.forName("bigdc.Comun.DimProducto").toDF
val FactCargo = DeltaTable.forName("bigdc.Facturacion.FactCargo").toDF
val DimContrato = DeltaTable.forName("bigdc.Comun.DimContrato").toDF
val DimCliente = DeltaTable.forName("bigdc.Comun.DimCliente").toDF

// COMMAND ----------

val FechaCierre = java.time.LocalDate.now.withDayOfMonth(1).minusDays(1).toString

// COMMAND ----------

val Comp_Pago_Gas = FactResumenCierreDia.alias("cd")
  .filter(
    $"cd.FechaCierre" <= lit(FechaCierre) &&
    $"cd.FechaCierre" === last_day($"cd.FechaCierre") &&
    $"cd.FechaCierre" >= last_day(add_months(lit(FechaCierre), -12)) &&
    $"cd.IdTipoProducto" === 7014
  )
  .groupBy($"cd.IdContrato")
  .agg(
    when(max($"cd.EdadMora") <= 30, lit("1-Pago A Tiempo"))
      .when(max($"cd.EdadMora") > 30 && max($"cd.EdadMora") <= 60, lit("2-Pago Atrasado a 60 Dias"))
      .when(max($"cd.EdadMora") > 60, lit("3-No Paga a 90 Dias"))
      .otherwise(lit("4-Otro")) as "Comport_Pago_Gas",
    sum(
      when($"cd.EdadMora" >= -1 && $"cd.EdadMora" <= 0, lit(1))
      .otherwise(lit(0))
    ) as "Edad0Gas",
    sum(
      when($"cd.EdadMora" >= 1 && $"cd.EdadMora" <= 30, lit(1))
      .otherwise(lit(0))
    ) as "Edad30Gas",
    sum(
      when($"cd.EdadMora" >= 31 && $"cd.EdadMora" <= 60, lit(1))
      .otherwise(lit(0))
    ) as "Edad60Gas",
    sum(
      when($"cd.EdadMora" >= 61 && $"cd.EdadMora" <= 90, lit(1))
      .otherwise(lit(0))
    ) as "Edad90Gas",
    sum(
      when($"cd.EdadMora" > 90, lit(1))
      .otherwise(lit(0))
    ) as "EdadM90Gas"
  )
  

// COMMAND ----------

val Info_Brilla = FactResumenCierreDia.alias("cd")
  .filter(
    $"cd.FechaCierre" <= lit(FechaCierre) &&
    $"cd.FechaCierre" === last_day($"cd.FechaCierre") &&
    $"cd.FechaCierre" >= last_day(add_months(lit(FechaCierre), -12)) &&
    $"cd.IdTipoProducto" === 7055
  )
  .groupBy($"cd.IdContrato", $"cd.FechaCierre")
  .agg(max($"cd.EdadMora") as "EM")

// COMMAND ----------

val Comp_Pago_Brilla = Info_Brilla.alias("ib")
  .groupBy($"ib.IdContrato")
  .agg(
    when(max($"ib.EM") <= 30, lit("1-Pago A Tiempo"))
      .when(max($"ib.EM") > 30 && max($"ib.EM") <= 60, lit("2-Pago Atrasado a 60 Dias"))
      .when(max($"ib.EM") > 60, lit("3-No Paga a 90 Dias"))
      .otherwise(lit("5-Otro")) as "Comport_Pago_Brilla",
    sum(
      when($"ib.EM" >= -1 && $"ib.EM" <= 0, lit(1))
      .otherwise(lit(0))
    ) as "Edad0Brilla",
    sum(
      when($"ib.EM" >= 1 && $"ib.EM" <= 30, lit(1))
      .otherwise(lit(0))
    ) as "Edad30Brilla",
    sum(
      when($"ib.EM" >= 31 && $"ib.EM" <= 60, lit(1))
    .otherwise(lit(0))
    ) as "Edad60Brilla",
    sum(
      when($"ib.EM" >= 61 && $"ib.EM" <= 90, lit(1))
      .otherwise(lit(0))
    ) as "Edad90Brilla",
    sum(
      when($"ib.EM" > 90, lit(1))
      .otherwise(lit(0))
    ) as "EdadM90Brilla"
  )

// COMMAND ----------

val InfoDiferidos = DimDiferido.alias("dd")
  .join(
    DimProducto.alias("dp"),
    $"dd.IdProducto" === $"dp.IdProducto",
    "inner"
  )
  .filter(
    $"dd.is_current" === 1 &&
    ($"dp.IdTipoProducto" === 7014 || $"dp.IdTipoProducto" === 7055) &&
    $"dd.SaldoPendiente" > 0
  )
  .groupBy($"dd.IdContrato")
  .agg(
    max(
      when($"dp.IdTipoProducto" === 7014, $"dd.NumeroCuotas")
      .otherwise(lit(0))
    ) as "PlazoGas",
    max(
      when($"dp.IdTipoProducto"===7014, $"dd.NumeroCuotas"-$"dd.NumeroCuotasFacturadas")
      .otherwise(lit(0))
    ) as "CuotasPendientesGas",
    sum(
      when($"dp.IdTipoProducto" === 7014, $"dd.ValorTotal")
      .otherwise(lit(0))
    ) as "SaldoInicialGas",
    max(
      when($"dp.IdTipoProducto" === 7055, $"dd.NumeroCuotas")
      .otherwise(lit(0))
    ) as "PlazoBrilla",
    max(
      when($"dp.IdTipoProducto" === 7055, $"dd.NumeroCuotas" -$"dd.NumeroCuotasFacturadas")
      .otherwise(lit(0))
    ) as "CuotasPendientesBrilla",
    sum(
      when($"dp.IdTIpoProducto" === 7055, $"dd.ValorTotal")
      .otherwise(lit(0))
    ) as "SaldoInicialBrilla"
  )

// COMMAND ----------

val rst = FactCargo.alias("fc")
  .join(
    DimProducto.alias("dp"),
    $"fc.IdProducto" === $"dp.IdProducto",
    "inner"
  )
  .filter(
    !$"fc.Programa".isin(20, 700, 2016) &&
    $"fc.TipoProceso" === "A" &&
    $"fc.Signo".isin("DB", "CR") &&
    $"fc.FechaContabilizacion" >= last_day(add_months(lit(FechaCierre), -12)) &&
    $"fc.FechaContabilizacion" <= date_add(last_day(lit(FechaCierre)), 1) &&
    $"dp.IdTipoProducto".isin(7014, 7055)
  )
  .groupBy(
    $"dp.IdContrato",
    $"fc.IdProducto",
    $"dp.IdTipoProducto",
    last_day($"fc.FechaContabilizacion" as "Periodo")
  )
  .agg(
    sum(
      when($"fc.Signo" === "DB", $"fc.Valor")
      .otherwise(-$"fc.Valor")
    ) as "Cuota"
    )

// COMMAND ----------

val CuotaFacturada = rst
  .groupBy($"IdContrato")
  .agg(
    avg(
      when($"IdTipoProducto" === 7014, $"Cuota")
      .otherwise(lit(0))
    ) as "CuotaGas",
    avg(
      when($"IdTipoProducto" === 7055, $"Cuota")
      .otherwise(lit(0))
    ) as "CuotaBrilla"
  )

// COMMAND ----------

val insumos = FactResumenCierreDia.alias("cd")
  .join(
    Comp_Pago_Gas.alias("cpg"),
    $"cd.IdContrato" === $"cpg.IdContrato",
    "left"
  )
  .join(
    Comp_Pago_Brilla.alias("cpb"),
    $"cd.IdContrato" === $"cpb.IdContrato",
    "left"
  )
  .join(
    InfoDiferidos.alias("di"),
    $"cd.IdContrato" === $"di.IdContrato",
    "left"
  )
  .join(
    DimContrato.alias("co"),
    $"cd.IdContrato" === $"co.IdContrato" &&
    $"cd.IdProducto" === $"co.IdProducto",
    "inner"
  )
  .join(
    DimProducto.alias("d"),
    $"cd.IdContrato" === $"d.IdContrato" &&
    $"cd.IdProducto" === $"d.IdProducto" &&
    $"d.IdContrato" === $"co.IdContrato",
    "inner"
  )
  .join(
    DimCliente.alias("c"),
    $"c.IdCliente" === $"co.IdCliente",
    "left"
  )
  .join(
    CuotaFacturada.alias("cf"),
    $"cd.IdContrato" === $"cf.IdContrato",
    "left"
  )
  .filter(
    $"FechaCierre" === lit(FechaCierre) &&
    $"cd.IdTipoProducto" === 7014 &&
    $"cd.IdCategoria" === 1 &&
    !$"cd.IdEstadoCorte".isin(110, 111, 112)
  )
  .select(
    $"c.TipoIdentificacion",
    $"cd.Identificacion",
    $"cd.IdContrato",
    $"cpg.Comport_Pago_Gas",
    $"cpb.Comport_Pago_Brilla",
    $"Edad0Gas".cast("int") as "Edad0Gas",
    $"Edad30Gas".cast("int") as "Edad30Gas",
    $"Edad60Gas".cast("int") as "Edad60Gas",
    $"Edad90Gas".cast("int") as "Edad90Gas",
    $"EdadM90Gas".cast("int") as "EdadM90Gas",
    $"Edad0Brilla".cast("int") as "Edad0Brilla",
    $"Edad30Brilla".cast("int") as "Edad30Brilla",
    $"Edad60Brilla".cast("int") as "Edad60Brilla",
    $"Edad90Brilla".cast("int") as "Edad90Brilla",
    $"EdadM90Brilla".cast("int") as "EdadM90Brilla",
    lit("Cupo Aprobado") as "Tipo_Cupo",
    $"cd.Departamento",
    $"cd.IdCategoria",
    $"cd.IdSubCategoria",
    $"di.CuotasPendientesGas",
    $"di.CuotasPendientesBrilla",
    $"cf.CuotaGas".cast("int") as "CuotaGas",
    $"cf.CuotaBrilla".cast("int") as "CuotaBrilla"
  )

// COMMAND ----------

insumos.write.mode("overwrite").saveAsTable("analiticagdc.brilla.insumosscoring")
