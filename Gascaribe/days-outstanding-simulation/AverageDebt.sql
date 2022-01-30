DECLARE @Fecha DATETIME = '2019-10-31';

SELECT AVG( DeudaDiferida + DeudaCorrienteNoVencida + DeudaCorrienteVencida ) AS DeudaPromedio
FROM Cartera.FactResumenCierreDia
WHERE FechaCierre = @Fecha
	AND EdadMora > 90
	AND TipoProducto = 7055
;
