# Databricks notebook source
import delta.tables
import psycopg2
from delta.tables import *
from pyspark.sql.functions import asc, desc

# COMMAND ----------

results = DeltaTable.forName(spark, 'analiticagdc.brilla.scoringFNB')\
    .toDF()\
    .orderBy(desc("FechaPrediccion"))\
    .dropDuplicates(["IdContrato"])\
    .select("IdContrato", "CupoAsignado", "Nodo", "Riesgo")

newColumns = ["contract_id", "assigned_quotas", "nodo", "risk_level"]

for i in range(len(results.columns)):
    results = results.withColumnRenamed(results.columns[i], newColumns[i])

# COMMAND ----------

user = dbutils.secrets.get(scope='gascaribe', key='sb-user')
password = dbutils.secrets.get(scope='gascaribe', key='sb-password')
host = dbutils.secrets.get(scope='gascaribe', key='sb-host')
port = dbutils.secrets.get(scope='gascaribe', key='sb-port')
database = dbutils.secrets.get(scope='gascaribe', key='sb-database')

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
    table="staging.bi_assigned_quotas", 
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
    INSERT INTO public.bi_assigned_quotas(contract_id, assigned_quota, nodo, risk_level, created_at, updated_at)
    SELECT
        contract_id,
        assigned_quota,
        nodo,
        risk_level,
        CURRENT_TIMESTAMP as created_at,
        CURRENT_TIMESTAMP
    FROM staging.bi_assigned_quotas
    ON CONFLICT(contract_id)
    DO UPDATE SET assigned_quota = EXCLUDED.assigned_quota,
                nodo = EXCLUDED.nodo,
                risk_level = EXCLUDED.risk_level,
                updated_at = current_timestamp;
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
