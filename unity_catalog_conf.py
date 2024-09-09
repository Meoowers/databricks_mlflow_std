# Databricks notebook source
# MAGIC %pip install "mlflow-skinny[databricks]>=2.4.1"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
catalog = "catalog_models"
schema = "default"
model_name = "random_forest_01"
mlflow.set_registry_uri("databricks-uc")
mlflow.register_model("runs:/364583f56fb541c9a206a664ff922625/model", f"{catalog}.{schema}.{model_name}")
