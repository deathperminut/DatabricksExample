# Databricks notebook source
# MAGIC %md
# MAGIC Lunes: 0
# MAGIC
# MAGIC Martes: 1
# MAGIC
# MAGIC ...
# MAGIC
# MAGIC Domingo: 6
# MAGIC
# MAGIC Festivos: 7

# COMMAND ----------

import os
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, DateType
import random
import scipy.stats as stats
from statsmodels.tsa.stattools import grangercausalitytests
from datetime import date,datetime,timedelta
import holidays
today = datetime.now()
tomorrow = today + timedelta(days=1)
today_dt = today.strftime("%d-%m-%Y")

import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

# COMMAND ----------

years = [2018,2019,2021,2022,2023,2024]
festivos = []

for year in years:
    colombia_holidays = holidays.Colombia(years=year)
    festivos += [x.strftime("%Y-%m-%d") for x in colombia_holidays.keys()]

# COMMAND ----------

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


query = 'SELECT * FROM ComercializacionML.DatosEDA'
query_tgi = 'SELECT * FROM ComercializacionML.DatosEDATGI'

df = spark.read \
    .format("com.databricks.spark.sqldw") \
    .option("url", sqlDwUrl) \
    .option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
    .option("forwardSparkAzureStorageCredentials", "true") \
    .option("maxStrLength", "1024" ) \
    .option("query", query) \
    .load()

df_tgi = spark.read \
    .format("com.databricks.spark.sqldw") \
    .option("url", sqlDwUrl) \
    .option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
    .option("forwardSparkAzureStorageCredentials", "true") \
    .option("maxStrLength", "1024" ) \
    .option("query", query_tgi) \
    .load()

rawData = df.toPandas()
rawData_tgi = df_tgi.toPandas()

# COMMAND ----------

rawData

# COMMAND ----------

ordinalNombre = {}
ordinalTipoUsuario = {}
for i,est in enumerate(rawData['NombreDispositivo'].unique()):
    ordinalNombre[est] = i

for i,tipo in enumerate(rawData['Tipo'].unique()):
    ordinalTipoUsuario[tipo] = i


ordinalNombre

# COMMAND ----------

def process_inputs(df,df_tgi):
    df = df.copy()
    df_tgi = df_tgi.copy()
    df_tgi = df_tgi[['NombreDispositivo', 'IdDispositivo', 'Tipo', 'FechaHistoria', 'VolumenM3']]
    df_tgi.columns = ['NombreDispositivo', 'IdDispositivo', 'Tipo', 'Fecha', 'VolumenM3']
    # Filtrar por tipo de usuario
    #df = df[~df['IdDispositivo'].isin(estacionesNoUtilizables)]

    index_to_change = df[df['NombreDispositivo'] == 'RELIGAMIENTO PEREIRA'].index

    df.loc[index_to_change, 'IdDispositivo'] = -23 



    newdf = df.groupby(['NombreDispositivo','IdDispositivo','Tipo','Fecha']).agg({'VolumenM3':['sum']}).reset_index()
    newdf.columns = newdf.columns.droplevel(1)

    fulldf = pd.concat([newdf,df_tgi],axis=0)

    for i,est in enumerate(fulldf['NombreDispositivo'].unique()):
        ordinalNombre[est] = i

    for i,tipo in enumerate(fulldf['Tipo'].unique()):
        ordinalTipoUsuario[tipo] = i
    
    ## Filtrar por tipo de usuario
    #df = df[df['TipoUsuario'] == tipoUsuario]
    
    festivosBin = []
    for fecha in fulldf['Fecha']:
        if str(fecha) in festivos:
            festivosBin.append(1)
        else:
            festivosBin.append(0)
            
    fulldf['Festivos'] = festivosBin
    
    festivosBinEspeciales = []
    for fecha in fulldf['Fecha']:
        if str(fecha) in festivosEspeciales:
            festivosBinEspeciales.append(1)
        else:
            festivosBinEspeciales.append(0)
            
    fulldf['FestivosEspeciales'] = festivosBinEspeciales
    
    # Cambiar fecha por datetime
    fulldf['Fecha'] = pd.to_datetime(fulldf['Fecha'])
    fulldf['DiaDeSemana'] = fulldf['Fecha'].apply(lambda x: x.dayofweek)

    for i,festivo in enumerate(fulldf['Festivos']):
        if festivo == 1:
            fulldf['DiaDeSemana'][i] = 8
        else:
            pass

    fulldf['OrdinalNombre'] = fulldf['NombreDispositivo'].replace(ordinalNombre).astype(int)
    fulldf['OrdinalTipoUsuario'] = fulldf['Tipo'].replace(ordinalTipoUsuario).astype(int)

    
    fulldf['VolumenM3'] = fulldf['VolumenM3'].astype('float')
    fulldf['Festivos'] = fulldf['Festivos'].astype('int')
    fulldf['FestivosEspeciales'] = fulldf['FestivosEspeciales'].astype('int')
    

    return fulldf
    

# COMMAND ----------

X = process_inputs(rawData,rawData_tgi)

# COMMAND ----------

X[X['Fecha'] == X['Fecha'].max()]

# COMMAND ----------

len(X[X['VolumenM3'] > 0])/len(X)

# COMMAND ----------

import pandas as pd
from scipy.stats import ttest_ind

def compareHolidays(df, consumo, day_column, holiday_column,TipoUsuario,soloConDiasDeSemana=True):
    df = df.dropna()
    # Filter data for holidays and normal days
    holidays = df[(df[holiday_column] == 1) & (df['Tipo'] == TipoUsuario)][consumo]
    if soloConDiasDeSemana:
        normal_days = df[(df[day_column].isin([1,2,3,4,5])) & (df['Tipo'] == TipoUsuario)][consumo]
    else:
        normal_days = df[(df[holiday_column] == 0) & (df['Tipo'] == TipoUsuario)][consumo]
    
    # t-test
    t_statistic, p_value = ttest_ind(holidays, normal_days)
    
    # Print the results
    print("Comparacion de consumo de gas:")
    print("----------------------------")
    print("Media de Consumo de Gas en Dia Festivo: {:.2f} m3".format(holidays.mean()))
    print("Media de Consumo de Gas en Dia Normal: {:.2f} m3".format(normal_days.mean()))
    print("T-Statistic: {:.2f}".format(t_statistic))
    print("p-Value: {:.4f}".format(p_value))
    
    # Interpret the results
    if p_value < 0.05:
        print("Hay diferencia estadisticamente significativa entre los dias festivos y los normales.")
    else:
        print("No hay diferencia estadisticamente significativa entre los dias festivos y los normales.")


# COMMAND ----------

for usuario in X['Tipo'].unique():
    print(usuario)
    compareHolidays(X,'VolumenM3','DiaDeSemana','Festivos',usuario,soloConDiasDeSemana=False)

# COMMAND ----------

X[X['VolumenM3'].isna()].reset_index(drop=True).iloc[0]['DiaDeSemana']

# COMMAND ----------

xd = X[X['VolumenM3'].isna()]

#xd[(xd['Nombre'] == 'SUAN') & (xd['Festivos'] == 1) & (xd[xd['DiaDeSemana'] == 2])]
xd.shape

# COMMAND ----------

def replaceNaN(df,festivos=True):
    df = df.copy()

    nanValues = df[df['VolumenM3'].isna()].reset_index(drop=True)
    nonnanValues = df[~df['VolumenM3'].isna()].reset_index(drop=True)

    for row,value in enumerate(nanValues['VolumenM3']):
        diaDeSemana = nanValues['DiaDeSemana'].iloc[row]
        estacion = nanValues['Nombre'].iloc[row]
        festivo = nanValues['Festivos'].iloc[row]
        if festivo:
            listOfValues = list(nonnanValues[(nonnanValues['Nombre'] == estacion) & (nonnanValues['Festivos'] == 1)].tail(4)['VolumenM3'])
            replaceValue = sum(listOfValues)/len(listOfValues)
        else:
            listOfValues = list(nonnanValues[(nonnanValues['Nombre'] == estacion) & (nonnanValues['Festivos'] == 0) & (nonnanValues['DiaDeSemana'] == diaDeSemana)].tail(4)['VolumenM3'])
            replaceValue = sum(listOfValues)/len(listOfValues)
        
        nanValues['VolumenM3'].iloc[row] = replaceValue
    

    newdf = pd.concat([nanValues,nonnanValues],axis=0)
    newdf = newdf.sort_values(by=['Nombre', 'Fecha'])

    return newdf

# COMMAND ----------

def replaceOutliers(df,festivos=True):
    df = df.copy()

    for est in df['Nombre'].unique():
        estacion = df[df['Nombre'] == est].reset_index(drop=True)
        

    nanValues = df[df['VolumenM3'].isna()].reset_index(drop=True)
    nonnanValues = df[~df['VolumenM3'].isna()].reset_index(drop=True)

    for row,value in enumerate(nanValues['VolumenM3']):
        diaDeSemana = nanValues['DiaDeSemana'].iloc[row]
        estacion = nanValues['Nombre'].iloc[row]
        festivo = nanValues['Festivos'].iloc[row]
        if festivo:
            listOfValues = list(nonnanValues[(nonnanValues['Nombre'] == estacion) & (nonnanValues['Festivos'] == 1)].tail(4)['VolumenM3'])
            replaceValue = sum(listOfValues)/len(listOfValues)
        else:
            listOfValues = list(nonnanValues[(nonnanValues['Nombre'] == estacion) & (nonnanValues['Festivos'] == 0) & (nonnanValues['DiaDeSemana'] == diaDeSemana)].tail(4)['VolumenM3'])
            replaceValue = sum(listOfValues)/len(listOfValues)
        
        nanValues['VolumenM3'].iloc[row] = replaceValue
    

    newdf = pd.concat([nanValues,nonnanValues],axis=0)
    newdf = newdf.sort_values(by=['Nombre', 'Fecha'])

    return newdf

# COMMAND ----------

import pandas as pd
from scipy.stats import ttest_ind

def compareHolidays(df, consumo, day_column, holiday_column,TipoUsuario,dia):
    df = df.dropna()
    # Filter data for holidays and normal days
    holidays = df[(df[holiday_column] == 1) & (df['TipoUsuario'] == TipoUsuario)][consumo]
    
    normal_days = df[(df['FestivosEspeciales'] == 1) & (df['TipoUsuario'] == TipoUsuario)][consumo]
    
    # t-test
    t_statistic, p_value = ttest_ind(holidays, normal_days)
    
    # Print the results
    print("Comparacion de consumo de gas:")  
    print("----------------------------")
    print("Media de Consumo de Gas en Dia Festivo: {:.2f} m3".format(holidays.mean()))
    print("Media de Consumo de Gas en Dia Normal: {:.2f} m3".format(normal_days.mean()))
    print("T-Statistic: {:.2f}".format(t_statistic))
    print("p-Value: {:.4f}".format(p_value))
    
    # Interpret the results
    if p_value < 0.05:
        print("Hay diferencia estadisticamente significativa entre los dias festivos y los normales.")
    else:
        print("No hay diferencia estadisticamente significativa entre los dias festivos y los normales.")


# COMMAND ----------

for usuario in X['TipoUsuario'].unique():
    print(usuario)
    compareHolidays(X,'VolumenM3','DiaDeSemana','Festivos',usuario,dia=0)

# COMMAND ----------

for usuario in X['TipoUsuario'].unique():
    print(usuario)
    compareHolidays(X,'VolumenM3','DiaDeSemana','FestivosEspeciales',usuario,dia=6)

# COMMAND ----------

X[X['DiaDeSemana'] == 0]

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

hm = {}
for estacion in X['NombreDispositivo'].unique():
    data = np.array(X[X['NombreDispositivo'] == estacion].dropna()['VolumenM3'])
    print(estacion)
    

    # Method 1: Visual Inspection
    # Histogram
    #plt.hist(data, bins=30, density=True)
    #plt.title("Histogram of the Data")
    #plt.show()

    # QQ Plot
    #stats.probplot(data, plot=plt)
    #plt.title("QQ Plot of the Data")
    #plt.show()

    # Box Plot
    #plt.boxplot(data)
    #plt.title("Box Plot of the Data")
    #plt.show()

    # Method 2: Descriptive Statistics
    mean = np.mean(data)
    median = np.median(data)
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)

    print(f"Mean: {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Skewness: {skewness:.2f}")
    print(f"Kurtosis: {kurtosis:.2f}")

    # Method 3: Shapiro-Wilk Test
    shapiro_stat, shapiro_p = stats.shapiro(data)
    print(f"Shapiro-Wilk Test - Statistic: {shapiro_stat:.4f}, p-value: {shapiro_p:.4f}")


    hm[estacion] = [shapiro_stat,shapiro_p]
    # Method 4: Anderson-Darling Test
    anderson_stat, anderson_crit_vals, anderson_sig_levels = stats.anderson(data)
    print(f"Anderson-Darling Test - Statistic: {anderson_stat:.4f}")
    print("Critical Values:")
    for crit, sig_level in zip(anderson_crit_vals, anderson_sig_levels):
        print(f"  {sig_level}%: {crit:.4f}")

    result = stats.anderson(data)

    for i in range(len(result.significance_level)):
        if result.statistic < result.critical_values[i]:
            print(f"At {result.significance_level[i]:.1f}% significance level, data is approximately normally distributed.")
        else:
            print(f"At {result.significance_level[i]:.1f}% significance level, data is not normally distributed.")

# COMMAND ----------

for key,val in hm.items():
    if val[1] >= 0.05:
        print(estacion)

# COMMAND ----------

import numpy as np

def hampel_filter(data, window_size=3, n_sigma=3):
    """
    Hampel filter to detect and replace outliers in a dataset.

    Parameters:
        data (numpy.ndarray): 1-dimensional array of data.
        window_size (int): Size of the window used to calculate the median and MAD.
        n_sigma (float): Number of standard deviations to set the threshold.

    Returns:
        numpy.ndarray: A copy of the data with outliers replaced by the median of non-outlying values.
    """
    # Make a copy of the data to avoid modifying the original array
    filtered_data = data.copy()
    
    # Calculate the number of data points in the dataset
    n = len(filtered_data)
    
    # Calculate the median and the Median Absolute Deviation (MAD)
    med = np.median(filtered_data)
    mad = np.median(np.abs(filtered_data - med))
    
    # Set the threshold for outliers
    threshold = n_sigma * 1.4826 * mad
    
    # Loop through each data point and replace outliers with the median
    for i in range(n):
        if np.abs(filtered_data[i] - med) > threshold:
            filtered_data[i] = med
    
    print(len(filtered_data))
    # Method 3: Shapiro-Wilk Test
    shapiro_stat, shapiro_p = stats.shapiro(filtered_data)
    print(f"Shapiro-Wilk Test - Statistic: {shapiro_stat:.4f}, p-value: {shapiro_p:.4f}")
    if shapiro_p >= 0.05:
        print('SII/')

    return filtered_data

# COMMAND ----------

for estacion in X['NombreDispositivo'].unique():
    data = np.array(X[X['NombreDispositivo'] == estacion].dropna()['VolumenM3'])
    hampel_filter(data, window_size=3, n_sigma=3)

# COMMAND ----------

data = X[X['NombreDispositivo'] == 'CITY GATE PEREIRA'].dropna()['VolumenM3']

plt.hist(data, bins=30, density=True)
plt.title("Histogram of the Data")
plt.show()

# COMMAND ----------

X

# COMMAND ----------

newX = X[X['Fecha'] >= '2022-08-04']

# COMMAND ----------

import scipy.stats as stats

def compareAmountOfDays(df):
    counter = 0
    bestDays = {}
    intervalDays = [4,6,10,20,30,52]
    for estacion in df['NombreDispositivo'].unique():
        estaciondf = df[df['NombreDispositivo'] == estacion]
        print(estacion)
        for day in estaciondf['DiaDeSemana'].unique():
            estaciondfDia = estaciondf[estaciondf['DiaDeSemana'] == day]
            meanDay = list(estaciondfDia['VolumenM3'].dropna())

            print(day)
            for days in intervalDays:
                meanInterval = list(estaciondfDia.tail(days)['VolumenM3'].dropna())
                t_stat, p_value = stats.ttest_ind(meanDay, meanInterval,nan_policy='omit')
                
                print(f'Mean for all days: {sum(meanDay)/len(meanDay)}')
                print(f'Mean for {days} days: {sum(meanInterval)/len(meanInterval)}')
                print(f'p_value: {p_value}')

                if p_value <= 0.05:
                    significant = True
                    bestDays[f'{estacion} {day}'] = [days,p_value]
                    counter += 1
                    break
                    
                else:
                    significant = False
    print(counter)
    return bestDays
                
bestdays = compareAmountOfDays(newX)

# COMMAND ----------

mostCommonValues = list(bestdays.values())
mostcommon = [mcv[0] for mcv in mostCommonValues]
max(set(mostcommon), key=mostcommon.count)

# COMMAND ----------

mostCommonValues

# COMMAND ----------

bestEstaciones = list(bestdays.keys())
set([b[:-2] for b in bestEstaciones])

# COMMAND ----------

import numpy as np

def hampel_filter(data, window_size, num_devs=3.0):
    """
    Apply Hampel filter to a 1D array of data.

    Parameters:
        data (numpy.ndarray): The 1D input array.
        window_size (int): The size of the window used for outlier detection.
        num_devs (float): Number of MADs to be used as a threshold for outlier detection.

    Returns:
        numpy.ndarray: The filtered data with outliers replaced by the median of the window.
    """
    filtered_data = data.copy()

    for i in range(len(data)):
        lower_bound = max(0, i - window_size)
        upper_bound = min(len(data), i + window_size)

        window = data[lower_bound:upper_bound]
        median = np.median(window)
        mad = np.median(np.abs(window - median))
        threshold = num_devs * 1.4826 * mad  # Factor of 1.4826 makes the MAD scale estimate consistent with std deviation

        if np.abs(data[i] - median) > threshold:
            filtered_data[i] = median


    print(len(filtered_data))
    # Method 3: Shapiro-Wilk Test
    shapiro_stat, shapiro_p = stats.shapiro(filtered_data)
    print(f"Shapiro-Wilk Test - Statistic: {shapiro_stat:.4f}, p-value: {shapiro_p:.4f}")
    if shapiro_p >= 0.05:
        print('SII')
    return filtered_data


# COMMAND ----------

for estacion in X['NombreDispositivo'].unique():
    data = np.array(X[X['NombreDispositivo'] == estacion].dropna()['VolumenM3'])
    hampel_filter(data, window_size=100)

# COMMAND ----------

def hampel_filter(df,window_size,n=4,num_devs=3.0):
    """
    Apply Hampel filter to a 1D array of data.

    Parameters:
        data (numpy.ndarray): The 1D input array.
        window_size (int): The size of the window used for outlier detection.
        num_devs (float): Number of MADs to be used as a threshold for outlier detection.

    Returns:
        numpy.ndarray: The filtered data with outliers replaced by the median of the window.
    """
    cols = list(df.columns) + ['ConsumoCorregido']
    newdf = pd.DataFrame(columns=cols)

    tomorrow_day = tomorrow.weekday()

    estaciones = df['NombreDispositivo'].unique()
    for estacion in estaciones:
        dataDF = df[df['NombreDispositivo'] == estacion].fillna(0).reset_index(drop=True)
        data = np.array(dataDF['VolumenM3'])

        filtered_data = data.copy()
        

        for i in range(len(data)):
            lower_bound = max(0, i - window_size)
            upper_bound = min(len(data), i + window_size)

            window = data[lower_bound:upper_bound]
            median = np.median(window)
            mad = np.median(np.abs(window - median))
            threshold = num_devs * 1.4826 * mad  # Factor of 1.4826 makes the MAD scale estimate consistent with std deviation

            if np.abs(data[i] - median) > threshold:
                day = dataDF.iloc[i]['DiaDeSemana']
                listOfValues = list(dataDF[dataDF['DiaDeSemana'] == day].tail(n)['VolumenM3'])
                meanValues = sum(listOfValues)/len(listOfValues)
                print(meanValues)
                filtered_data[i] = meanValues
                print(f'Indice reemplazado: {i}')

        dataDF['ConsumoCorregido'] = filtered_data

        newdf = pd.concat([dataDF,newdf],axis=0)

    return newdf
    

# COMMAND ----------

_ = hampel_filter(X,window_size=100)

# COMMAND ----------

_[_['VolumenM3'] != _['ConsumoCorregido']].shape[0]/_.shape[0]

# COMMAND ----------

_[_['VolumenM3'] != _['ConsumoCorregido']]

# COMMAND ----------

for usuario in _['TipoUsuario'].unique():
    print(usuario)
    compareHolidays(_,'VolumenM3','DiaDeSemana','Festivos',usuario,dia=0)

# COMMAND ----------

_[_['FestivosEspeciales'].shift(-1) == 1]

# COMMAND ----------

import pandas as pd
from scipy.stats import ttest_ind

def compareHolidays(df, consumo, day_column, holiday_column,TipoUsuario,dia):
    df = df.dropna()

    diasdesemana = {'Martes': 1,
                    'Domingo': 6}

    # Filter data for holidays and normal days
    holidays = df[(df['DiaDeSemana'] == diasdesemana[dia]) & (df['Tipo'] == TipoUsuario)][consumo]
    #print(diasdesemana[dia])
    lunesFestivos = list(set(df[(df['Festivos'] == 1) & (df['DiaDeSemana'] == 0)]['Fecha']))
    
    if dia == 'Martes':
        normal_days = df[(df['Fecha'].shift(1).isin(lunesFestivos)) & (df['Tipo'] == TipoUsuario)][consumo]
    else:
        normal_days = df[(df['Fecha'].shift(-1).isin(lunesFestivos)) & (df['Tipo'] == TipoUsuario)][consumo]
    #print(normal_days)
    # t-test
    t_statistic, p_value = ttest_ind(holidays, normal_days)
    
    # Print the results
    print("Comparacion de consumo de gas:")  
    print("----------------------------")
    print("Media de Consumo de Gas en Dia Festivo: {:.2f} m3".format(holidays.mean()))
    print("Media de Consumo de Gas en Dia Normal: {:.2f} m3".format(normal_days.mean()))
    print("T-Statistic: {:.2f}".format(t_statistic))
    print("p-Value: {:.4f}".format(p_value))
    
    # Interpret the results
    if p_value < 0.05:
        print(f"Hay diferencia estadisticamente significativa entre los dias Domingo antes de festivo y los normales.")
    else:
        print("No hay diferencia estadisticamente significativa entre los dias Domingo antes de festivo y los normales.")


# COMMAND ----------


lunesFestivos = list(set(_[(_['Festivos'] == 1) & (_['DiaDeSemana'] == 0)]['Fecha']))
_[(_['Fecha'].shift(-1).isin(lunesFestivos)) | (_['Fecha'].shift(1).isin(lunesFestivos))]

# COMMAND ----------

for usuario in X['Tipo'].unique():
    print(usuario)
    compareHolidays(X,'VolumenM3','DiaDeSemana','Festivos',usuario,dia='Domingo')

# COMMAND ----------


