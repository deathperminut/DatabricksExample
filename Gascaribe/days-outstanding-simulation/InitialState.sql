DECLARE @Fecha DATETIME = '2019-10-31';

SELECT CASE WHEN re.EdadMora <= 0 THEN 0 WHEN re.EdadMora > 90 THEN 120 ELSE CEILING( re.EdadMora / 30.0 ) * 30.0 END AS RangoEdadMora, COUNT(1) AS Cantidad
FROM Cartera.FactResumenCierreDia re
WHERE re.FechaCierre = @Fecha AND re.TipoProducto = 7055
GROUP BY CASE WHEN re.EdadMora <= 0 THEN 0 WHEN re.EdadMora > 90 THEN 120 ELSE CEILING( re.EdadMora / 30.0 ) * 30.0 END
ORDER BY CASE WHEN re.EdadMora <= 0 THEN 0 WHEN re.EdadMora > 90 THEN 120 ELSE CEILING( re.EdadMora / 30.0 ) * 30.0 END
;
