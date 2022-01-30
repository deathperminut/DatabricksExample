/*
 * To avoid the Inspection Paradox (http://allendowney.blogspot.com/2015/08/the-inspection-paradox-is-everywhere.html) we need to sample suspensions
 * from the suspension table, and not by taking all suspended users at a single time. This is to avoid undersampling users with shorter suspension times.
 * Specifically, we take all suspensions in a given interval and randomly choose a number of days since the user was suspended and query the user's state
 * at that point in time. This is to equally represent all possible states for a given suspension length.
 */

SET NOCOUNT ON;

DECLARE @FechaInicioHistoria DATETIME = '2018-07-01';
DECLARE @FechaLimiteHistoria DATETIME = '2019-07-01';
DECLARE @FechaLimiteSuspensiones DATETIME = '2019-08-01';

WITH
CuotaFacturada AS (
    SELECT Producto
        ,SUM( CASE WHEN Signo IN ( 'DB', 'SA', 'AP' ) THEN Valor ELSE -Valor END ) AS Cuota
    FROM Facturacion.FactCargo
    WHERE TipoProceso = 'A'
        AND FechaContabilizacion >= @FechaInicioHistoria
        AND FechaContabilizacion < @FechaLimiteHistoria
    GROUP BY Producto
),
IncidenciaMora AS (
    SELECT Producto
        ,SUM( CASE WHEN EdadMora BETWEEN 1 AND 30 THEN 1 ELSE 0 END ) AS Veces30
        ,SUM( CASE WHEN EdadMora BETWEEN 31 AND 60 THEN 1 ELSE 0 END ) AS Veces60
        ,SUM( CASE WHEN EdadMora BETWEEN 61 AND 90 THEN 1 ELSE 0 END ) AS Veces90
        ,SUM( CASE WHEN EdadMora > 90 THEN 1 ELSE 0 END ) AS VecesMas90
        ,MAX( EdadMora ) AS MoraMaxima
    FROM Cartera.FactResumenCierreDia
    WHERE FechaCierre = EOMONTH( FechaCierre )
        AND FechaCierre >= @FechaInicioHistoria
        AND FechaCierre < @FechaLimiteHistoria
    GROUP BY Producto
),
SuspensionesReconexiones AS (
    SELECT pr.IdContrato AS Contrato
        ,COUNT( DISTINCT su.FechaAplicacion ) AS Suspensiones
        ,COUNT( DISTINCT CASE WHEN su.FechaReconexion <> '1900-01-01' AND su.FechaReconexion < @FechaLimiteHistoria THEN su.FechaReconexion ELSE NULL END ) AS Reconexiones
    FROM Comun.FactSuspension su
    INNER JOIN Comun.DimProducto pr
        ON su.IdProducto = pr.IdProducto
    WHERE su.FechaAplicacion >= @FechaInicioHistoria
        AND su.FechaAplicacion < @FechaLimiteHistoria
    GROUP BY pr.IdContrato
),
SuspensionesCartera AS (
    SELECT
        CASE
            WHEN su.FechaReconexion = '1900-01-01'
            THEN DATEADD(
                DAY,
                FLOOR( RAND( su.IdSuspension ) * DATEDIFF( DAY, su.FechaAplicacion, ( SELECT Hoy FROM Comun.Hoy ) ) ),
                CAST( su.FechaAplicacion AS DATE )
            )
            WHEN DATEDIFF( DAY, su.FechaAplicacion, su.FechaReconexion ) <= 0
            THEN CAST( su.FechaAplicacion AS DATE )
            ELSE DATEADD(
                DAY,
                FLOOR( RAND( su.IdSuspension ) * DATEDIFF( DAY, su.FechaAplicacion, su.FechaReconexion ) ),
                CAST( su.FechaAplicacion AS DATE )
            )
        END AS FechaCierre
        ,su.IdProducto
        ,su.FechaAplicacion
        ,su.FechaReconexion
    FROM Comun.FactSuspension su
    WHERE su.FechaAplicacion >= @FechaLimiteHistoria
        AND su.FechaAplicacion < @FechaLimiteSuspensiones
        AND su.IdTipoSuspension = 2
)
SELECT re.FechaCierre
    ,re.Producto
    ,re.EdadMora
    ,CASE WHEN re.EdadMora = -1 THEN 0 ELSE CEILING( re.EdadMora / 30.0 ) * 30.0 END AS RangoEdadMora
    ,re.Subcategoria AS Estrato
    ,re.DeudaCorrienteNoVencida
    ,re.DeudaCorrienteVencida
    ,re.DeudaDiferida
    ,re.CantRefiUltimoAño
    ,re.CantHistoriaRefi
    ,CASE WHEN rf.Producto IS NULL THEN 'NO' ELSE 'SI' END AS Refinanciado
    ,im.Veces30
    ,im.Veces60
    ,im.Veces90
    ,im.VecesMas90
    ,im.MoraMaxima
    ,ISNULL( sr.Suspensiones, 0 ) AS Suspensiones
    ,ISNULL( sr.Reconexiones, 0 ) AS Reconexiones
    ,ISNULL( cu.Cuota, 0 ) AS Cuota
    ,DATEDIFF( DAY, CAST( su.FechaAplicacion AS DATE ), re.FechaCierre ) AS DiasSuspendido
    ,CASE
        WHEN su.FechaReconexion <> '1900-01-01' AND su.FechaAplicacion > su.FechaReconexion THEN 0
        WHEN su.FechaReconexion <> '1900-01-01' THEN DATEDIFF( DAY, su.FechaAplicacion, su.FechaReconexion )
        ELSE -1
    END AS TiempoTotalSuspension
FROM SuspensionesCartera su
INNER JOIN Cartera.FactResumenCierreDia re
    ON re.FechaCierre = su.FechaCierre
    AND re.Producto = su.IdProducto
LEFT JOIN Cartera.DimRefinanciadosRecuperacion rf
    ON rf.FechaCierre = re.FechaCierre
    AND re.Producto = rf.Producto
INNER JOIN IncidenciaMora im
    ON re.Producto = im.Producto
LEFT JOIN SuspensionesReconexiones sr
    ON re.Contrato = sr.Contrato
LEFT JOIN CuotaFacturada cu
    ON re.Producto = cu.Producto
WHERE re.FechaCierre >= @FechaLimiteHistoria
;
