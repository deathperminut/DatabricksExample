# Databricks notebook source
import pandas as pd
from fbprophet import Prophet


# COMMAND ----------
df = pd.read_csv(f"/Volumes/archive/gdcba/ecl/etapas.csv")

# Preprocess the data
df['FechaCierre'] = pd.to_datetime(df['FechaCierre'])
df.sort_values('FechaCierre', inplace=True)


# COMMAND ----------
# Create a new DataFrame for each category
categories = df['TipoProducto'].unique().tolist() + df['Clasificacion'].unique().tolist()
dataframes = {category: df[df['TipoProducto'] == category] if category in df['TipoProducto'].unique() else df[df['Clasificacion'] == category] for category in categories}


# COMMAND ----------
# Fit a Prophet model on each DataFrame and make a forecast
forecasts = {}
for category, dataframe in dataframes.items():
    model = Prophet()
    model.fit(dataframe.rename(columns={'FechaCierre': 'ds', 'Cartera': 'y'}))
    future = model.make_future_dataframe(periods=60)  # forecast the next two months
    forecast = model.predict(future)
    forecasts[category] = forecast