# Databricks notebook source
import psycopg2
from delta.tables import *
from pyspark.sql.functions import asc, desc

# COMMAND ----------

# Constants
dwDatabase = dbutils.secrets.get(scope='efigas', key='dwh-name')
dwServer = dbutils.secrets.get(scope='efigas', key='dwh-host')
dwUser = dbutils.secrets.get(scope='efigas', key='dwh-user')
dwPass = dbutils.secrets.get(scope='efigas', key='dwh-pass')
dwJdbcPort = dbutils.secrets.get(scope='efigas', key='dwh-port')
dwJdbcExtraOptions = ""
sqlDwUrl = "jdbc:sqlserver://" + dwServer + ".database.windows.net:" + dwJdbcPort + ";database=" + dwDatabase + ";user=" + dwUser + ";password=" + dwPass + ";" + dwJdbcExtraOptions
storage_account_name = dbutils.secrets.get(scope='efigas', key='bs-name')
blob_container = dbutils.secrets.get(scope='efigas', key='bs-container')
blob_storage = storage_account_name + ".blob.core.windows.net"
config_key = "fs.azure.account.key."+storage_account_name+".blob.core.windows.net"
blob_access_key = dbutils.secrets.get(scope='efigas', key='bs-access-key')
spark.conf.set(config_key, blob_access_key)

# COMMAND ----------

# Data Ingestion
query = """
    SELECT *
    FROM ComercializacionML.IngestaBricks  
    """

deltaDF = spark.read \
  .format("com.databricks.spark.sqldw") \
  .option("url", sqlDwUrl) \
  .option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
  .option("forwardSparkAzureStorageCredentials", "true") \
  .option("maxStrLength", "1024" ) \
  .option("query", query) \
  .load()

# COMMAND ----------

deltaDF.write.mode("overwrite").saveAsTable("analiticaefg.comercializacion.factvolumenreal")

# COMMAND ----------

user = dbutils.secrets.get(scope='efigas', key='com-user')
password = dbutils.secrets.get(scope='efigas', key='com-password')
host = dbutils.secrets.get(scope='efigas', key='com-host')
port = dbutils.secrets.get(scope='efigas', key='com-port')
database = dbutils.secrets.get(scope='efigas', key='com-database')

# COMMAND ----------

results = DeltaTable.forName(spark, 'analiticaefg.comercializacion.factvolumenreal').toDF()\
    .select("idestacion", "fecha", "volumen")

# COMMAND ----------

# Define the connection properties
url = f"jdbc:postgresql://{host}:{port}/{database}"
properties = {
    "driver": "org.postgresql.Driver",
    "user": user,
    "password": password
}

# COMMAND ----------

results.write.jdbc(
    url, 
    table="public.volumenes_bi", 
    mode="overwrite", 
    properties=properties
)

# COMMAND ----------

def pgConnection(user, password, host, port, db):
    """
    It creates a connection to a PostgreSQL database and returns the connection and cursor objects

    :param user: The username to connect to the database
    :param password: The password for the user
    :param host: the hostname of the server
    :param port: 5432
    :param db: The name of the database to connect to
    :return: A connection and a cursor.
    """

    connection = psycopg2.connect(
        user=user,
        password=password,
        host=host,
        port=port,
        database=db,
        connect_timeout=60)
    cursor = connection.cursor()
    return connection, cursor

# COMMAND ----------

upsert_statement = """
INSERT INTO volumenes(id_estacion, volumen, energia, fecha, created_at, updated_at)
SELECT
    idestacion AS id_estacion,
    volumen,
    NULL AS energia,
    fecha + INTERVAL '5' HOUR AS fecha,
    current_timestamp AS created_at,
    current_timestamp AS updated_at
FROM volumenes_bi
ON CONFLICT (id_estacion, fecha)
DO UPDATE SET
    volumen = EXCLUDED.volumen,
    updated_at = EXCLUDED.updated_at
"""

# COMMAND ----------

cur = None
conn = None

try:
    conn, cur = pgConnection(user, password, host, port, database)

    try:
        cur.execute(upsert_statement)
        conn.commit()

    except Exception as e:
        print(f"Upsert failed. {e}")

except Exception as e:
    print(f"Database connection failed. {e}")

finally:
    if(cur):
        cur.close()
    if(conn):
        conn.close()
