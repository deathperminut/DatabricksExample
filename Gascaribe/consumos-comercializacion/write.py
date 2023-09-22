# Databricks notebook source
import psycopg2
from delta.tables import *
from pyspark.sql.functions import asc, desc

# COMMAND ----------

user = dbutils.secrets.get(scope='gascaribe', key='com-user')
password = dbutils.secrets.get(scope='gascaribe', key='com-password')
host = dbutils.secrets.get(scope='gascaribe', key='com-host')
port = dbutils.secrets.get(scope='gascaribe', key='com-port')
database = dbutils.secrets.get(scope='gascaribe', key='com-database')

# COMMAND ----------
results = DeltaTable.forName(spark, 'analiticagdc.comercializacion.factconsumoproyectado')\
    .select("id", "valorpoyectado", "fecha")

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
    table="staging.estaciones", 
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
