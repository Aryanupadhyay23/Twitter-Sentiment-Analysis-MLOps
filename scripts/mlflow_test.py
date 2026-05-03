import mlflow
import dagshub
import random

dagshub.init(
    repo_owner="Aryanupadhyay23",
    repo_name="Twitter-Sentiment-Analysis-MLOps",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/Aryanupadhyay23/Twitter-Sentiment-Analysis-MLOps.mlflow"
)

# FIX: set experiment
mlflow.set_experiment("test-experiment")

with mlflow.start_run(run_name="test_run"):
    mlflow.log_param("param1", random.randint(0, 100))
    mlflow.log_metric("metric1", random.random())
    mlflow.set_tag("tag1", "test_tag")

print("MLflow test run completed successfully.")