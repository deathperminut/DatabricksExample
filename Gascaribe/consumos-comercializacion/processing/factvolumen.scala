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

val inicio = lit("2023-01-01").cast("date")
val fin = (current_timestamp()-expr("INTERVAL 29 HOURS")).cast("date")

val fechas = DeltaTable.forName("bigdc.comun.dimfecha").toDF
	.filter($"Fecha".between(inicio, fin))
	.select($"Fecha")

val dispositivos = DeltaTable.forName("analiticagdc.comercializacion.dimhomologaciondispositivos").toDF
	.filter($"IdDispositivo".contains("4-"))

val base = DeltaTable.forName("bigdc.novo.dimclasificaciondispositivos").toDF.as("c")
	.join(
		dispositivos
		.select($"IdDispositivo")
		.distinct
		.as("d"),
		Seq("IdDispositivo"),
		"inner")
	.join(
		fechas.as("f"),
		$"f.Fecha".between($"c.FechaInicio", date_add($"c.FechaFin", -1)),
		"inner")
	.select(
		$"c.IdDispositivo",
		$"c.IdHistoriaDispositivo",
		$"c.IdColumna",
		$"f.Fecha",
		row_number.over(Window.partitionBy($"c.IdDispositivo", $"f.Fecha").orderBy($"c.IdColumna".desc))
		.as("Medidor"))

val faltantes = dispositivos
	.join(
		base,
		Seq("IdDispositivo"),
		"leftanti")
	.select(
		$"IdComercializacion",
		$"IdDispositivo")
	.distinct

val volumen = base.as("b")
	.join(
		dispositivos.as("c"),
		Seq("IdDispositivo", "Medidor"),
		"inner")
	.join(
		DeltaTable.forName("bigdc.novo.facthistoriadispositivo").toDF.as("f"),
		date_trunc("day", $"FechaHoraHistoria") === $"b.Fecha" and
		$"b.IdHistoriaDispositivo" === $"f.IdHistoriaDispositivo" and
		$"b.IdColumna" === $"f.IdColumna" and
		$"f.AplicadoDeltaMode" === 1 and
		$"f.is_current" === 1,
		"left")
	.select(
		$"c.IdComercializacion".cast(IntegerType),
		$"b.Fecha".cast(DateType),
		when(
			$"IdDispositivo".isin("4-335-1-2", "4-1026-1-2"),
			coalesce($"f.ValorColumna", lit(0)) / 35.3147)
		.otherwise(coalesce($"f.ValorColumna", lit(0)))
		.cast(DecimalType(20,4))
		.as("Volumen"),
		row_number.over(Window.partitionBy($"c.IdComercializacion", $"b.Fecha").orderBy($"f.datalake_at".desc)).as("rn"))
	.filter($"rn" === 1)
	.drop("rn")

val volumenFaltantes = faltantes.as("f")
	.join(
		DeltaTable.forName("bigdc.novo.dimhistoriadispositivo").toDF.as("dh"),
		$"f.IdDispositivo" === $"dh.IdDispositivo" and
		$"dh.is_current" === 1 and
		lower($"dh.Nombre").contains("daily") and
		lower($"dh.Nombre").contains("single"),
		"inner")
	.crossJoin(fechas)
	.join(
		DeltaTable.forName("bigdc.novo.facthistoriadispositivo").toDF.as("fh"),
		date_trunc("day", $"fh.FechaHoraHistoria") === $"Fecha" and
		$"dh.IdHistoriaDispositivo" === $"fh.IdHistoriaDispositivo" and
		$"fh.IdColumna" === 2 and
		$"fh.AplicadoDeltaMode" === 1 and
		$"fh.is_current" === 1,
		"left")
	.select(
		$"f.IdComercializacion".cast(IntegerType),
		$"Fecha".cast(DateType),
		coalesce($"fh.ValorColumna", lit(0))
		.cast(DecimalType(20,4)).as("Volumen"),
		row_number.over(Window.partitionBy($"f.IdComercializacion", $"Fecha").orderBy($"fh.datalake_at".desc)).as("rn"))
	.filter($"rn" === 1)
	.drop("rn")

// COMMAND ----------

// DBTITLE 1,Sobreescribir tabla 
volumen.union(volumenFaltantes).write.mode("overwrite").saveAsTable("analiticagdc.comercializacion.factvolumen")
