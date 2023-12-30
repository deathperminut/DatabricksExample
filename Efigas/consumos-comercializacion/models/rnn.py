# Databricks notebook source
import os
import pandas as pd
import numpy as np
import pickle as pkl
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, DateType
from pyspark.sql import SparkSession
from pyspark.sql.functions import max as pySparkMax
from pyspark.sql.functions import col
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from delta.tables import DeltaTable
import random
import time
from datetime import date,datetime,timedelta,timezone
today = datetime.now(timezone(timedelta(hours=-5), 'EST'))
yesterday = (datetime.now(timezone(timedelta(hours=-5), 'EST')) - timedelta(1))
tomorrow = (datetime.now(timezone(timedelta(hours=-5), 'EST')) + timedelta(1))
two_days_prior = (datetime.now(timezone(timedelta(hours=-5), 'EST')) - timedelta(2)).strftime("%Y-%m-%d")
today_dt = today.strftime("%Y-%m-%d")
one_year_ago = (datetime.now(timezone(timedelta(hours=-5), 'EST')) - timedelta(days=365)).strftime("%Y-%m-%d")
two_years_ago = (datetime.now(timezone(timedelta(hours=-5), 'EST')) - timedelta(days=730)).strftime("%Y-%m-%d")


import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

storage_account_name = dbutils.secrets.get(scope='efigas', key='bs-name')
blob_container = dbutils.secrets.get(scope='efigas', key='bs-container')
blob_storage = storage_account_name + ".blob.core.windows.net"
config_key = "fs.azure.account.key."+storage_account_name+".blob.core.windows.net"
blob_access_key = dbutils.secrets.get(scope='efigas', key='bs-access-key')
spark.conf.set(config_key, blob_access_key)

# COMMAND ----------

insumo = DeltaTable.forName(spark, "analiticaefg.comercializacion.insumo").toDF()
estado = DeltaTable.forName(spark, "analiticaefg.comercializacion.estado").toDF()

t1 = estado.groupBy("estacion").agg(pySparkMax("fecharegistro").alias("latest_fecharegistro"))

estado_join = t1.alias('t1').join(estado.alias('e'), (t1.estacion == estado.estacion) & (t1.latest_fecharegistro == estado.fecharegistro),'inner').select('t1.estacion','e.estado','t1.latest_fecharegistro')

results = insumo.alias('i').join(estado_join.alias('e'), (insumo.estacion == estado_join.estacion),'inner').select("i.estacion", "i.fecha", "i.volumenm3", "i.festivos", "i.diadesemana", "i.volumen_corregido", "residuals", "e.estado","e.latest_fecharegistro").where('e.estado = "Activa"').orderBy('i.estacion','i.fecha')

results = results.toPandas()

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import LabelEncoder

def train_model_4(estacion,runScheduler,learningRateDecay,runEarlyStopping,patience,epochs=200):
    
    newdf = results[results['estacion'] == estacion]
    data = newdf['volumen_corregido'].values

    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))

    sequence_length = 10
    train_size = int(len(data) * 0.95)

    def create_sequences(data, seq_length):
        sequences = []
        targets = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i+seq_length])
            targets.append(data[i+seq_length])
        return np.array(sequences), np.array(targets)

    train_data = normalized_data[:train_size]
    test_data = normalized_data[train_size:]

    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)


    model = Sequential([
            LSTM(units=512, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            Bidirectional(LSTM(units=128)),
            Dense(units=64, activation='relu'),
            Dense(units=1)
            ])

    model.compile(optimizer='adam', loss='mean_squared_error',run_eagerly=True)

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)

    def scheduler(epoch, lr):
        if epoch < 5:
            return lr
        else:
            return lr * tf.math.exp(learningRateDecay)

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

    if runScheduler:
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2,callbacks=[lr_scheduler])
    elif runEarlyStopping:
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2,callbacks=[early_stopping])
    else:
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2)

    return model,history,X_test,y_test,data


# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

historical_data = results.copy()

yesterday_date = datetime.now() - timedelta(days=1)
tomorrow_date = datetime.now() + timedelta(days=1)

def predict_consumption_for_date(historical_data, target_date, train_date):
    unique_stations = historical_data['estacion'].unique()
    predictions = pd.DataFrame(columns=['estacion', 'fecha', 'prediccion'])
    test_predictions_df = pd.DataFrame(columns=['estacion', 'fecha', 'prediccion'])
    historical_data['fecha'] = pd.to_datetime(historical_data['fecha'])

    for station in unique_stations:
        #historical_data['fecha'] = pd.to_datetime(historical_data['fecha'])
        station_data = historical_data[(historical_data['estacion'] == station) & (historical_data['fecha'] < target_date)]
        consumption_data = station_data['volumen_corregido']

        
        sequence_length = 10

        if len(consumption_data) < sequence_length:
            continue  

        
        last_sequence = consumption_data.tail(sequence_length).values

        
        X_pred = last_sequence.reshape((-1, sequence_length, 1))

        model, history, X_test, y_test, data = train_model_4(station, False, 0.1, False, 10, 50)
        
        prediction = model.predict(X_pred)

        
        min_value = consumption_data.min()
        max_value = consumption_data.max()

        predicted_value = prediction * (max_value - min_value) + min_value

        
        pred_df = pd.DataFrame({
            'estacion': [station],
            'fecha': [target_date.strftime('%Y-%m-%d')],
            'prediccion': predicted_value.flatten()  
        })

        predictions = pd.concat([predictions, pred_df], ignore_index=True)


        # 

        test_df = consumption_data.iloc[-(sequence_length+1):-1].values
        test_values = test_df.reshape((-1, sequence_length, 1))

        test_predictions = model.predict(test_values)
        test_predictions = test_predictions * (max_value - min_value) + min_value

        test_pred_df = pd.DataFrame({
            'estacion': [station],
            'fecha': [train_date.strftime('%Y-%m-%d')],
            'prediccion': test_predictions.flatten()  
        })

        test_predictions_df = pd.concat([test_predictions_df, test_pred_df], ignore_index=True)


    return predictions, test_predictions_df



predictions_tomorrow, predictions_yesterday = predict_consumption_for_date(historical_data, tomorrow_date, yesterday.date())


# def get_yesterday_prediction_with_true_value(historical_data, target_date, train_date):
#     yesterday_predictions = predict_consumption_for_date(historical_data, target_date, train_date)

#     true_values_yesterday = historical_data[historical_data['fecha'] == target_date]
#     true_values_yesterday = true_values_yesterday.rename(columns={'volumen_corregido': 'valor_real'})

#     yesterday_predictions['fecha'] = pd.to_datetime(yesterday_predictions['fecha'])
#     true_values_yesterday['fecha'] = pd.to_datetime(true_values_yesterday['fecha'])
    
#     yesterday_predictions_with_true_value = pd.merge(yesterday_predictions, true_values_yesterday, on=['estacion', 'fecha'], how='inner')
#     return yesterday_predictions_with_true_value, yesterday_predictions, true_values_yesterday

# yesterday_predictions_with_true_value, _yp, _tvp = get_yesterday_prediction_with_true_value(historical_data, yesterday.date(),yesterday.date())

# print("Tomorrow's Predictions:")
# print(predictions_tomorrow)

# print("\nYesterday's Predictions with True Values:")
# print(yesterday_predictions_with_true_value)


# COMMAND ----------

import time

time.sleep(1000000)

# COMMAND ----------

predictions_yesterday['fecha'] = pd.to_datetime(predictions_yesterday['fecha'])
results['fecha'] = pd.to_datetime(results['fecha'])

# COMMAND ----------

preds_yesterday = predictions_yesterday.merge(results, on=['fecha','estacion'], how='inner')
preds_yesterday['error_absoluto'] = abs((preds_yesterday['volumenm3'] - preds_yesterday['prediccion'])*100/preds_yesterday['volumenm3'])
preds_yesterday = preds_yesterday[['estacion','error_absoluto']]
preds_yesterday.head()

# COMMAND ----------

predicciones = predictions_tomorrow.merge(preds_yesterday,on='estacion',how='inner')
predicciones['estado'] = 'Activa'
predicciones['modelo'] = 'RNN'
predicciones['fecha'] = pd.to_datetime(predicciones['fecha'])
predicciones = predicciones[['estacion','fecha','prediccion','error_absoluto','modelo','estado']]
predicciones = predicciones.rename(columns={'prediccion':'predicciones'})

# COMMAND ----------

predicciones

# COMMAND ----------

schema = StructType([
    StructField("estacion", StringType(), True),
    StructField("fecha", DateType(), True),
    StructField("predicciones", FloatType(), True),
    StructField("error_absoluto", FloatType(), True),
    StructField("modelo", StringType(), True),
    StructField("estado", StringType(), True)
    ])
df = spark.createDataFrame(predicciones, schema = schema)

deltaTable = DeltaTable.forName(spark, 'analiticaefg.comercializacion.predicciones_rnn')

deltaTable.alias("t").merge(
    df.alias("s"),
    "t.estacion = s.estacion AND t.fecha = s.fecha"
).whenNotMatchedInsertAll().execute()
