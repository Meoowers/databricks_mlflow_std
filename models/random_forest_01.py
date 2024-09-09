# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Import libs

# COMMAND ----------

import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

# COMMAND ----------

# MAGIC %md
# MAGIC # 02 - Set the experiment id, and MLFLOW config

# COMMAND ----------

mlflow.set_experiment(experiment_id="3392864180412703")
mlflow.autolog()
db = load_diabetes()

# COMMAND ----------

# MAGIC %md
# MAGIC # 03 - Test/train split and model training

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
rf.fit(X_train, y_train)


# COMMAND ----------

# MAGIC %md
# MAGIC # 04 - Predict

# COMMAND ----------

# Use the model to make predictions on the test dataset.
predictions = rf.predict(X_test)

# COMMAND ----------


