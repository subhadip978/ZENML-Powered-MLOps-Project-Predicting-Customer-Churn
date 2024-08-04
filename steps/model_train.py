import logging
import pandas as pd 
from zenml import step
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
from src.model_dev import (
   
    LinearRegressionModel,
    
)

from zenml.client import Client
import mlflow

experiment_tracker=Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def train_model(X_train:pd.DataFrame,
                X_test:pd.DataFrame,
                y_train: pd.Series,
                y_test: pd.Series,
                config: ModelNameConfig,)-> RegressorMixin:

    try:
          
        model=None
        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            
            model = LinearRegressionModel()
            trained_model=model.train(X_train,y_train)
            return trained_model
        
        else:
            raise ValueError("model {} not supported".format(config.model_name))
        


    except Exception as e:
        logging.error("error in model training:{}".format(e))
        raise e


