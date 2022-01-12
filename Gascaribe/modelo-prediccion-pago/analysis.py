# Databricks notebook source
import os
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_theme(style="darkgrid")
import matplotlib.pyplot as plt
import time 
from pprint import pprint
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from scipy import stats
from scipy.stats import skew 
from scipy.stats import norm
from imblearn.pipeline import Pipeline, make_pipeline

from sklearnex import patch_sklearn
patch_sklearn()

import sklearn
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn import metrics
from sklearn.utils import shuffle, resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix

from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# COMMAND ----------

# Constants
dwDatabase = os.environ.get("DWH_NAME")
dwServer = os.environ.get("DWH_HOST")
dwUser = os.environ.get("DWH_USER")
dwPass = os.environ.get("DWH_PASS")
dwJdbcPort = os.environ.get("DWH_PORT")
dwJdbcExtraOptions = ""
sqlDwUrl = "jdbc:sqlserver://" + dwServer + ".database.windows.net:" + dwJdbcPort + ";database=" + dwDatabase + ";user=" + dwUser + ";password=" + dwPass + ";" + dwJdbcExtraOptions
storage_account_name = os.environ.get("BS_NAME")
blob_container = os.environ.get("BS_CONTAINER")
blob_storage = storage_account_name + ".blob.core.windows.net"
config_key = "fs.azure.account.key."+storage_account_name+".blob.core.windows.net"
blob_access_key = os.environ.get("BS_ACCESS_KEY")
spark.conf.set(config_key, blob_access_key)
is_training = dbutils.widgets.get("is_training") == "true"

# COMMAND ----------

# Data Ingestion
query = 'SELECT * FROM ModeloPrediccionPago.DatosModelo WHERE Pago_Actual IN ' + ('(0, 1)' if is_training else '(-1)')

df = spark.read \
  .format("com.databricks.spark.sqldw") \
  .option("url", sqlDwUrl) \
  .option("tempDir", "wasbs://" + blob_container + "@" + storage_account_name + ".blob.core.windows.net/") \
  .option("forwardSparkAzureStorageCredentials", "true") \
  .option("maxStrLength", "1024" ) \
  .option("query", query) \
  .load()
raw_data = df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA

# COMMAND ----------

dataset = raw_data.copy()

# COMMAND ----------

dataset.drop(['FechaEntrenamiento','FechaPrediccion','Producto'],axis=1,inplace=True)

# COMMAND ----------

columns = ['EdadMora_30','EdadMora_30','EdadMora_60','EdadMora_60','Pago','Pago_Temprano']
for c in columns :
    dataset[c] = dataset[c].astype(float)

# COMMAND ----------

columns = ['Categoria','Refinanciado','EstadoFinanciero','EstadoProducto']
for c in columns :
    dataset[c] = dataset[c].astype(str)

# COMMAND ----------

dataset

# COMMAND ----------

dataset[['Pago_Actual']].groupby(['Pago_Actual']).size() 

# COMMAND ----------

dataset[['Pago_Actual']].groupby(['Pago_Actual']).size() / dataset.shape[0]

# COMMAND ----------

#All variables correlation
correlation_matrix = dataset.corr()
plt.figure(figsize=(8,8),facecolor='white')
sns.heatmap(correlation_matrix,square=True,annot=True,cmap="RdYlGn")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fixing Features

# COMMAND ----------

numerical_features = dataset.dtypes[dataset.dtypes != 'object'].index
numerical_features

# COMMAND ----------

categorical_features = dataset.dtypes[dataset.dtypes == 'object'].index
categorical_features

# COMMAND ----------

#Normalize numercial features
numerical = dataset[numerical_features]
scaler = MinMaxScaler()
d = scaler.fit_transform(numerical)
scaled_df = pd.DataFrame(d, columns=numerical_features)
for c in numerical_features :
    dataset[c] = scaled_df[c]

# COMMAND ----------

#One-hot encoding
one_hot_columns = ['Categoria','Refinanciado','EstadoFinanciero','EstadoProducto']
categories = [[101,102,103,104,105,106,201,202], [1,0], [1,2,3,4], ['Activo','Suspendido','Retirado sin instalación','Retirado']]
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, categories=categories)
ds_encoded = pd.DataFrame(encoder.fit_transform(dataset[one_hot_columns]))
ds_encoded.columns = encoder.get_feature_names(one_hot_columns)
dataset.drop(one_hot_columns, axis=1, inplace=True)
dataset = pd.concat([dataset, ds_encoded], axis=1)

# COMMAND ----------

dataset

# COMMAND ----------

dataset.columns

# COMMAND ----------

# MAGIC %md
# MAGIC # Modeling

# COMMAND ----------

training_data, testing_data = train_test_split(dataset, test_size=0.3, random_state=25)

# COMMAND ----------

print(training_data.shape)
print(testing_data.shape)

# COMMAND ----------

train_X = training_data.drop(['Pago_Actual'], axis = 1)
train_Y = training_data['Pago_Actual']
test_X = testing_data.drop(['Pago_Actual'], axis = 1)
test_Y = testing_data['Pago_Actual']

# COMMAND ----------

#Oversampling
X = train_X.copy()
Y = train_Y.copy()
ros = RandomOverSampler(random_state=123,sampling_strategy = 0.8)
X, Y = ros.fit_resample(X, Y)
train_X = X.copy()
train_Y = Y.copy()

# COMMAND ----------

#Prints model accuracy and plots ROC curve
def test_model(model):
    start_time = time.time()
    model.fit(train_X,train_Y)
    prediction = model.predict(test_X)
    print('accuracy',metrics.accuracy_score(prediction,test_Y))
    print("--- %s seconds ---" % (time.time() - start_time))
    y_pred_proba = model.predict_proba(test_X)[:,1]
    fpr, tpr, _ = metrics.roc_curve(test_Y,  y_pred_proba)
    auc = metrics.roc_auc_score(test_Y, y_pred_proba)
    plt.plot(fpr,tpr,label="data, auc="+str(auc))
    plt.legend(loc=4)
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Logistic Regression

# COMMAND ----------

model = LogisticRegression()
test_model(model)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random Forest

# COMMAND ----------

model=RandomForestClassifier(n_estimators=100)
test_model(model)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Adaboost

# COMMAND ----------

model=AdaBoostClassifier(n_estimators=100)
test_model(model)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cross Validation

# COMMAND ----------

X = dataset.drop(['Pago_Actual'], axis = 1)
Y = dataset['Pago_Actual']

# COMMAND ----------

stratified_kfold = StratifiedKFold(n_splits=3,random_state=123,shuffle=True)

# COMMAND ----------

def test_model_cv(model) :
    start_time = time.time()
    steps = Pipeline(steps = [['oversampler', RandomOverSampler(random_state=123,sampling_strategy = 0.8)],
                              ['classifier', model]])
    cv_result = cross_val_score(steps, X, Y, cv = stratified_kfold, scoring = "accuracy")
    print("--- %s seconds ---" % (time.time() - start_time))
    print('mean ',cv_result.mean())
    print('std ',cv_result.std())
    print('acurrancy ',cv_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Logistic Regression

# COMMAND ----------

model = LogisticRegression()
test_model_cv(model)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random Forest

# COMMAND ----------

model = RandomForestClassifier(n_estimators=100)
test_model_cv(model)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Adaboost

# COMMAND ----------

model = AdaBoostClassifier(n_estimators=100)
test_model_cv(model)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grid Search

# COMMAND ----------

#Prints model accuracy and plots ROC curve
def test_grid(model):
    start_time = time.time()
    prediction = model.predict(test_X)
    print('accuracy',metrics.accuracy_score(prediction,test_Y))
    print("--- %s seconds ---" % (time.time() - start_time))
    y_pred_proba = model.predict_proba(test_X)[:,1]
    fpr, tpr, _ = metrics.roc_curve(test_Y,  y_pred_proba)
    auc = metrics.roc_auc_score(test_Y, y_pred_proba)
    plt.plot(fpr,tpr,label="data, auc="+str(auc))
    plt.legend(loc=4)
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random Forest

# COMMAND ----------

param_grid = {
    'bootstrap': [True, False],
    'max_depth': [80, 90, 100, 110],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [200, 300, 400]
}

# COMMAND ----------

model = RandomForestClassifier()

# COMMAND ----------

grid_search = HalvingGridSearchCV(estimator = model, param_grid = param_grid, cv = stratified_kfold, verbose = 1)

# COMMAND ----------

grid_search.fit(train_X, train_Y)

# COMMAND ----------

grid_search.best_params_
# {'bootstrap': True,
#  'max_depth': 110,
#  'min_samples_leaf': 3,
#  'min_samples_split': 8,
#  'n_estimators': 300}

# COMMAND ----------

best_grid = grid_search.best_estimator_
grid_accuracy = test_grid(best_grid)

# COMMAND ----------

params = {'bootstrap': True,
        'max_depth': 110,
        'min_samples_leaf': 3,
        'min_samples_split': 8,
        'n_estimators': 300}
model = RandomForestClassifier(**params)
test_model_cv(model)
