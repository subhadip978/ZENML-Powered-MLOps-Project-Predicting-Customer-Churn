import logging
import pandas as pd
import numpy as np
from typing import Tuple
from zenml import step

from typing_extensions import Annotated

from src.data_cleaning import (
    DataCleaning,
    DataDivideStrategy,
    DataPreProcessStrategy,
)



@step
def clean_df(df:pd.DataFrame)->Tuple[Annotated[np.ndarray,"X_train"],
                                     Annotated[np.ndarray,"X_test"],
                                    Annotated[pd.Series,"y_train"],
                                    Annotated[pd.Series,"y_test"],]:
     
    try:
        preprocess_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("data cleaning complete")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(e)
        raise e