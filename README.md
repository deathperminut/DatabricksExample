# business-analytics
Proyectos de analítica del área de BI

En este repositorio se encuentran los proyectos de Machine Learning desarrollados por el equipo de BI.
El repositorio se integra directamente con Azure Databricks, que es el motor sobre el cual corren los modelos.

How to deploy?

Para desplegar los modelos, es necesario llevar los notebooks a la rama master y asegurar de que el notebook tenga un cluster asociado en Databricks.
Una vez el modelo está deplegado, se usa Azure DataFactory para llamarlo periódicamente.