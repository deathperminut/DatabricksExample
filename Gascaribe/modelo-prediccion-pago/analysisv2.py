# Databricks notebook source
import os
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_theme(style="darkgrid")
import matplotlib
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.inspection import permutation_importance
from sklearn import metrics
from sklearn.utils import shuffle, resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
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

# COMMAND ----------

# Data Ingestion
query = "SELECT * FROM ModeloPrediccionPago.DatosEntrenamiento"

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

dataset.isnull().sum(axis = 0)

# COMMAND ----------

dataset['AVGEdadPago'] = dataset['AVGEdadPago'].fillna(360)
dataset['AVGEdadMora'] = dataset['AVGEdadMora'].fillna(-360)

# COMMAND ----------

dataset.drop(['Producto'], axis=1, inplace=True)

# COMMAND ----------

# dataset.drop(dataset[dataset['EdadMora']>120].index, axis=0, inplace=True)
# dataset.reset_index(drop=True,inplace=True)

# COMMAND ----------

columns_numeric = ['SP30D','SP60D','SP90D','SP90M','AV','DV','V30D','V60D','V90D','V90M','AVGEdad','AVGEdadPago',
                   'AVGEdadMora','CuentasSaldadas','CantidadCuentas']
for c in columns_numeric :
    dataset[columns_numeric] = dataset[columns_numeric].astype(float)
    
columns_categorical = ['Categoria','TipoProducto','Refinanciado']
for c in columns_categorical :
    dataset[columns_categorical] = dataset[columns_categorical].astype(str)

# COMMAND ----------

dataset.describe()

# COMMAND ----------

print(dataset[['Pago']].groupby('Pago').size())
print(dataset[['Pago']].groupby('Pago').size() / dataset.shape[0])

# COMMAND ----------

print("Cuentas Antes del Vencimiento " + str(dataset['AV'].sum() / dataset['CuentasSaldadas'].sum()))
print("Cuentas Despues del Vencimiento " + str(dataset['DV'].sum() / dataset['CuentasSaldadas'].sum()))

# COMMAND ----------

plt.figure(figsize=(10, 5))
ax = sns.countplot(x='Pago',data=dataset)
ax.bar_label(ax.containers[0])
plt.show()

# COMMAND ----------

plt.figure(figsize=(10, 5))
ax = sns.countplot(x='Categoria',data=dataset)
ax.bar_label(ax.containers[0])
plt.show()

# COMMAND ----------

plt.figure(figsize=(10, 5))
ax = sns.countplot(x='EdadMora',data=dataset[dataset['EdadMora']<=390])
ax.bar_label(ax.containers[0])
plt.show()

# COMMAND ----------

plt.figure(figsize=(10, 5))
ax = sns.countplot(x='Refinanciado',data=dataset)
ax.bar_label(ax.containers[0])
plt.show()

# COMMAND ----------

col = ['SP30D','SP60D','SP90D','SP90M','AV','DV','V30D','V60D','V90D','V90M',]
ds_sum = dataset[col].sum()
fig, ax = plt.subplots(figsize=(15,5))
plt.bar(ds_sum.index,ds_sum)
ax.bar_label(ax.containers[0])
plt.show()

# COMMAND ----------

plt.figure(figsize=(10, 5))
ax = sns.countplot(x='CantidadCuentas',data=dataset)
ax.bar_label(ax.containers[0])
plt.show()

# COMMAND ----------

columns = ['SP30D','SP60D','SP90D','SP90M','AV','DV','V30D','V60D','V90D','V90M','CuentasSaldadas']
for c in columns:
    dataset['Ratio'+c] = dataset[c] / dataset['CantidadCuentas']
    
columns = ['AV','DV','V30D','V60D','V90D','V90M']
for c in columns:
    dataset['Ratio'+c+'Saldada'] = dataset[c] / dataset['CuentasSaldadas']
    dataset['Ratio'+c+'Saldada'] = dataset['Ratio'+c+'Saldada'].fillna(0)

# COMMAND ----------

# #All variables correlation
# correlation_matrix = dataset.corr()
# plt.figure(figsize=(15,15),facecolor='white')
# sns.heatmap(correlation_matrix,square=True,annot=True,cmap="RdYlGn")

# COMMAND ----------

#Most correlated variables
correlation_matrix = dataset.corr()
most_correlated = correlation_matrix[abs(correlation_matrix['Pago']) >= 0.2]
plt.figure(figsize=(10,10),facecolor='white')
sns.heatmap(most_correlated[most_correlated.index],square=True,annot=True,cmap="RdYlGn")
plt.show()

# COMMAND ----------

final_features = np.concatenate((np.array(most_correlated.index),
               np.array(dataset.dtypes[dataset.dtypes == 'object'].index)),
               axis=0)

# COMMAND ----------

dataset = dataset[final_features]

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
one_hot_columns = ['Categoria','TipoProducto','Refinanciado']
categories = [[101,102,103,104,105,106,201,202],
              [-1,3,6121,7014,7052,7053,7054,7055],
              [1,0]]
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, categories=categories)
ds_encoded = pd.DataFrame(encoder.fit_transform(dataset[one_hot_columns]))
ds_encoded.columns = encoder.get_feature_names(one_hot_columns)
dataset.drop(one_hot_columns, axis=1, inplace=True)
dataset = pd.concat([dataset, ds_encoded], axis=1)

# COMMAND ----------

dataset.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # Modeling

# COMMAND ----------

X = dataset.drop(['Pago'], axis = 1)
Y = dataset['Pago']

# COMMAND ----------

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3)

# COMMAND ----------

#Oversampling
X = train_X.copy()
Y = train_Y.copy()
ros = RandomOverSampler(random_state=123,sampling_strategy = 1)
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
    print('precision',metrics.precision_score(prediction,test_Y))
    print('recall',metrics.recall_score(prediction,test_Y))
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

lr = LogisticRegression(random_state=123)
test_model(lr)

# COMMAND ----------

feature_importance=pd.DataFrame({'feature':list(train_X.columns),'feature_importance':[abs(i) for i in lr.coef_[0]]})
feature_importance[feature_importance['feature_importance']>0].sort_values('feature_importance',ascending=False)[:10]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random Forest

# COMMAND ----------

rf=RandomForestClassifier(random_state=123,n_estimators=100,n_jobs=-1)
test_model(rf)

# COMMAND ----------

feature_importances=pd.DataFrame({'features':train_X.columns,'feature_importance':rf.feature_importances_})
feature_importance[feature_importance['feature_importance']>0].sort_values('feature_importance',ascending=False)[:10]

# COMMAND ----------

# MAGIC %md
# MAGIC ### AdaBoost

# COMMAND ----------

abc=AdaBoostClassifier(random_state=123,n_estimators=100)
test_model(abc)

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLPClassifier

# COMMAND ----------

mlp = MLPClassifier(random_state=123)
test_model(mlp)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cross Validation

# COMMAND ----------

X = dataset.drop(['Pago'], axis = 1)
Y = dataset['Pago']

# COMMAND ----------

stratified_kfold = StratifiedKFold(n_splits=3,random_state=123,shuffle=True)

# COMMAND ----------

def test_model_cv(model) :
    start_time = time.time()
    steps = Pipeline(steps = [['oversampler', RandomOverSampler(random_state=123,sampling_strategy = 1)],
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

lr = LogisticRegression()
test_model_cv(lr)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random Forest

# COMMAND ----------

rf = RandomForestClassifier(n_jobs=-1)
test_model_cv(rf)

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLPClassifier

# COMMAND ----------

mlp = MLPClassifier()
test_model_cv(mlp)

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
# MAGIC ### MLPClassifier

# COMMAND ----------

parameter_grid = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

# COMMAND ----------

model = MLPClassifier(max_iter=100)

# COMMAND ----------

grid_search = HalvingGridSearchCV(estimator = model, param_grid = parameter_grid, cv = stratified_kfold, verbose = 1, n_jobs=-1)
grid_search.fit(train_X, train_Y)

# COMMAND ----------

grid_search.best_params_
# {'activation': 'tanh',
#  'alpha': 0.001,
#  'hidden_layer_sizes': (100,),
#  'learning_rate': 'adaptive',
#  'solver': 'adam'}

# COMMAND ----------

best_grid = grid_search.best_estimator_
grid_accuracy = test_grid(best_grid)

# COMMAND ----------

params = {'activation': 'tanh',
 'alpha': 0.001,
 'hidden_layer_sizes': (100,),
 'learning_rate': 'adaptive',
 'solver': 'adam'}
model = MLPClassifier(**params)
test_model_cv(model)

# COMMAND ----------

model.fit(train_X,train_Y)

# COMMAND ----------

y_true, y_pred = test_Y , model.predict(test_X)
from sklearn.metrics import classification_report
print('Results on the test set:')
print(classification_report(y_true, y_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random Forest

# COMMAND ----------

parameter_grid = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000]}

# COMMAND ----------

model = RandomForestClassifier(n_jobs=-1)

# COMMAND ----------

grid_search = HalvingGridSearchCV(estimator = model, param_grid = parameter_grid, cv = stratified_kfold, verbose = 1, n_jobs=-1)
grid_search.fit(train_X, train_Y)

# COMMAND ----------

grid_search.best_params_
# {'bootstrap': True,
#  'max_depth': 10,
#  'max_features': 'auto',
#  'min_samples_leaf': 4,
#  'min_samples_split': 10,
#  'n_estimators': 800}

# COMMAND ----------

best_grid = grid_search.best_estimator_
grid_accuracy = test_grid(best_grid)

# COMMAND ----------

params = {'bootstrap': True,
 'max_depth': 10,
 'max_features': 'auto',
 'min_samples_leaf': 4,
 'min_samples_split': 10,
 'n_estimators': 800}
model = RandomForestClassifier(**params)
test_model_cv(model)

# COMMAND ----------

model.fit(train_X,train_Y)

# COMMAND ----------

y_true, y_pred = test_Y , model.predict(test_X)
from sklearn.metrics import classification_report
print('Results on the test set:')
print(classification_report(y_true, y_pred))
