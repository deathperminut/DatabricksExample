# Databricks notebook source
from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup

import mlflow
import mlflow.sklearn
from mlflow.tracking.client import MlflowClient
from mlflow.models.signature import infer_signature

import sklearn
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from mlflow.utils.environment import _mlflow_conda_env

import cloudpickle

import pandas as pd
import numpy as np

# COMMAND ----------

fs = feature_store.FeatureStoreClient()

customer_features_df = fs.read_table(
  name='efg_segmentacion_features',
)

# COMMAND ----------

feature_lookups = [
    FeatureLookup(
      table_name = 'efg_segmentacion_features',
      lookup_key = 'Identificacion',
      feature_names = ["Recency-Score", "Monetary-Score", "Frequency-Score"]
    ),
  ]

training_set = fs.create_training_set(
    customer_features_df[["Identificacion"]],
    feature_lookups = feature_lookups,
    label = None,
    exclude_columns = ['Identificacion']
)

training_df = training_set.load_df().toPandas()

n_clusters = 5

kmeans = KMeans(n_clusters=n_clusters, random_state=0)

model = kmeans.fit(training_df)

centroids = model.cluster_centers_

params = {"n_clusters": n_clusters, "random_state": 0, "centroids": centroids}

#mlflow.log_metric("Centroid RMSD", np.sqrt(model.inertia_/len(training_df)))
#mlflow.log_params(params)

class Clasificador(mlflow.pyfunc.PythonModel):
    def __init__(self, trained_model):
        self.model = trained_model
 
    def preprocess_result(self, model_input):
        return model_input
 
    def postprocess_result(self, results):
        """Return post-processed results.
        Creates a set of fare ranges
        and returns the predicted range."""
        colors = ['#DF2020', '#81DF20', '#2095DF','#F4D03F','#C800FE']
        centroids = self.model.cluster_centers_
        clusters = pd.DataFrame(centroids, columns=['Recency-Score','Monetary-Score','Frequency-Score'])
        clusters['cluster'] = self.model.predict(clusters[['Recency-Score','Monetary-Score','Frequency-Score']]) 
        clusters['magnitude'] = np.sqrt(((clusters['Recency-Score']**2) + (clusters['Monetary-Score']**2) + (clusters['Frequency-Score']**2)))

        clusters['name'] = [0,0,0,0,0]
        clusters['name'].iloc[clusters['magnitude'].idxmax()] = 'Diamante'
        clusters['name'].iloc[clusters['magnitude'].idxmin()] = 'Hibernando'
        clusters['name'].iloc[clusters['magnitude'] == list(clusters['magnitude'].nsmallest(2))[1]] = 'Primiparos con plata'
        clusters['name'][(clusters['Frequency-Score'].isin(list(clusters['Frequency-Score'].nlargest(2)))) & (clusters['magnitude'].isin(list(clusters['magnitude'].nsmallest(4))))] = 'No puedo perder'
        clusters['name'].iloc[clusters['name'] == 0] = 'Primiparo'

        respuesta = pd.DataFrame(results, columns=['cluster'])

        Merged = respuesta.merge(clusters[['cluster','name']],on='cluster',how='left')

        return Merged[["name", "cluster"]]
 
    def predict(self, context, model_input):
        processed_df = self.preprocess_result(model_input.copy())
        results = self.model.predict(processed_df)
        return self.postprocess_result(results)

pyfunc_model = Clasificador(model)

with mlflow.start_run():
    fs.log_model(
        pyfunc_model,
        "RFM_efg",
        flavor=mlflow.pyfunc,
        training_set=training_set,
        registered_model_name="RFM_efg",
    )
    RMSD = np.sqrt(model.inertia_/len(training_df))
    mlflow.log_metric("Centroid RMSD", RMSD)
    mlflow.log_params(params)



# COMMAND ----------

from mlflow.tracking import MlflowClient
def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
      version_int = int(mv.version)
      if version_int > latest_version:
        latest_version = version_int
    return latest_version

# COMMAND ----------

client = MlflowClient()
if( RMSD < 1.35 and RMSD >1):
    version = get_latest_model_version("RFM_efg")
    client.transition_model_version_stage(
        name="RFM_efg",
        version=version,
        stage="Production",
        archive_existing_versions = True
    )
