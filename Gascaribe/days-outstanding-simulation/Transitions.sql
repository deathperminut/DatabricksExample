DECLARE @Fecha DATETIME = '2019-10-31';

WITH Base AS (
    SELECT cm.FechaCierre AS FechaCierreInicial
        ,re.FechaCierre AS FechaCierreFinal
        ,CASE WHEN cm.EdadMora <= 0 THEN 0 WHEN cm.EdadMora > 90 THEN 120 ELSE CEILING( cm.EdadMora / 30.0 ) * 30.0 END AS RangoEdadMoraInicial
        ,CASE WHEN re.EdadMora <= 0 THEN 0 WHEN re.EdadMora > 90 THEN 120 ELSE CEILING( re.EdadMora / 30.0 ) * 30.0 END AS RangoEdadMoraFinal
        ,COUNT( 1 ) AS Cantidad
    FROM Cartera.FactResumenCierreDia re
    INNER JOIN Cartera.FactResumenCierreDia cm
        ON re.Producto = cm.Producto
        AND re.FechaCierre = EOMONTH( cm.FechaCierre, 1 )
    WHERE re.FechaCierre = EOMONTH( re.FechaCierre )
        AND cm.FechaCierre = EOMONTH( cm.FechaCierre )
        AND cm.FechaCierre <= @Fecha
        AND cm.FechaCierre > EOMONTH( @Fecha, -9 )
        AND cm.TipoProducto = 7055
    GROUP BY cm.FechaCierre
        ,re.FechaCierre
        ,CASE WHEN cm.EdadMora <= 0 THEN 0 WHEN cm.EdadMora > 90 THEN 120 ELSE CEILING( cm.EdadMora / 30.0 ) * 30.0 END
        ,CASE WHEN re.EdadMora <= 0 THEN 0 WHEN re.EdadMora > 90 THEN 120 ELSE CEILING( re.EdadMora / 30.0 ) * 30.0 END
)
SELECT FechaCierreInicial
    ,FechaCierreFinal
    ,RangoEdadMoraInicial
    ,RangoEdadMoraFinal
    ,Cantidad
    ,SUM( Cantidad ) OVER( PARTITION BY FechaCierreInicial, FechaCierreFinal, RangoEdadMoraInicial ) AS Total
FROM Base
ORDER BY FechaCierreInicial
    ,FechaCierreFinal
    ,RangoEdadMoraInicial
    ,RangoEdadMoraFinal
;
