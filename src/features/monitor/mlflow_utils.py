# src/features/monitor/mlflow_utils.py
import mlflow
import os

class MLflowTracker:
    """Experiment tracking utilities for logging parameters, metrics, and artifacts."""
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
    
    def log_params(self, params: dict):
        """Logs a dictionary of parameters."""
        for param_key, param_value in params.items():
            mlflow.log_param(param_key, param_value)
            
    def log_metrics(self, metrics: dict, step: int = None):
        """Logs multiple metrics at a particular step."""
        for metric_key, metric_value in metrics.items():
            mlflow.log_metric(metric_key, metric_value, step=step)
            
    def log_metric(self, name: str, value: float, step: int = None):
        """Logs a single metric."""
        mlflow.log_metric(name, value, step=step)
        
    def log_artifact(self, file_path: str):
        """Logs a specific artifact file."""
        if os.path.exists(file_path):
            mlflow.log_artifact(file_path)
        else:
            print(f"Warning: Artifact {file_path} not found.")

def start_run(run_name: str = None):
    return mlflow.start_run(run_name=run_name)
