# Business Analytics

This repository contains Machine Learning projects developed by the BA team. The repository is directly integrated with Azure Databricks, which is the engine on which the models run.

## Deployment

### Orchestrated by Azure Data Factory

To deploy the models, you need to move the notebooks to the master branch and ensure that the notebook has a cluster associated in Databricks. Once the model is deployed, use Azure DataFactory call it periodically using pipelines and the Databricks Linked Service.

### Orchestrated by Databricks Workflows

To deploy the models, you need to move the notebooks to the master branch and set up a job in Databricks Workflows to call it periodically or on demand.

## Usage

To use the models, you need to have access to the Azure Databricks workspace and the necessary credentials to run the notebooks. Once you have access, you can use the Databricks workspace to develop notebooks.

## Contributing

If you want to contribute to this repository, please follow the guidelines in the CONTRIBUTING.md file.