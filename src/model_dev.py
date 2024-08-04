import logging
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression



class Model(ABC):

    def train(self,x_train,y_train):
        pass


  


class LinearRegressionModel(Model):

    def train(self, X_train, y_train,**kwargs):
        try:

            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("model train complete")
            return reg
        
        except Exception as e:
            logging.error("error in training model:{}".formate(e))
            raise e

    # # For linear regression, there might not be hyperparameters that we want to tune, so we can simply return the score
    # def optimize(self, trial, x_train, y_train, x_test, y_test):
    #     reg = self.train(x_train, y_train)
    #     return reg.score(x_test, y_test)
