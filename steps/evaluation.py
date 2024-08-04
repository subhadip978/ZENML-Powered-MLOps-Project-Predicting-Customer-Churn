import logging


import mlflow
import numpy as np
import pandas as pd
from src.evaluation import MSE, RMSE, R2Score
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from typing import Tuple
from zenml import step
from zenml.client import Client

experiment_tracker=Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model:RegressorMixin,X_test:pd.DataFrame, y_test:pd.Series)->Tuple[
    Annotated[float,"r2_score"],
    Annotated[float,"rmse"],]:

    try:
        prediction = model.predict(X_test)

      
        mse_class = MSE()
        mse = mse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("mse",mse)
        

        r2_class = R2Score()
        r2_score = r2_class.calculate_score(y_test, prediction)
        logging.info(f'r2 score is {r2_score}')
        mlflow.log_metric("r2_score",r2_score)

        # Using the RMSE class for root mean squared error calculation
        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test, prediction)
        logging.info(f'rmse score is {rmse}')
                                          
        mlflow.log_metric("rmse",rmse)
        
        return r2_score, rmse

    except  Exception as e:
        logging.error("error in evaluate_model :{}".format(e))
        raise e