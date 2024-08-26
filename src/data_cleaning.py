import pandas as pd 
import numpy as np
import logging
import pickle
from typing import Union
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from abc import ABC,abstractmethod

from sklearn.model_selection import train_test_split



class DataStrategy(ABC):

    @abstractmethod
    def handle_data(self,data:pd.DataFrame)->Union[pd.DataFrame,pd.Series]:
        pass




class DataPreProcessStrategy(DataStrategy):

    def handle_data(self,df:pd.DataFrame)->pd.DataFrame:

        try:
            df = df.drop_duplicates(subset=['order_id','order_purchase_timestamp','product_id','customer_id','review_comment_message'])

            df.drop(['order_id','product_id','seller_id','customer_unique_id'], axis=1, inplace=True)
            # data=data.drop(
            #     [

                    
            #         "order_approved_at",
            #         "order_delivered_carrier_date",
            #         "order_delivered_customer_date",
            #         "order_estimated_delivery_date",
            #         "order_purchase_timestamp",
            #     ],
            #     axis=1
            # )
            df.dropna(subset=['shipping_limit_date','order_purchase_timestamp','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date'], inplace=True)
            intermediate_time = df['order_delivered_customer_date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").date()) - df['order_purchase_timestamp'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").date())
            df['purchase_delivery_difference'] = intermediate_time.apply(lambda x:x.days)

            df.dropna(subset=['shipping_limit_date','order_purchase_timestamp','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date'], inplace=True)
            intermediate_time = df['order_estimated_delivery_date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").date()) - df['order_delivered_customer_date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").date())
            df['estimated_actual_delivery_difference'] = intermediate_time.apply(lambda x:x.days)
            df = df[df['order_status'] != 'canceled']

            df['review_score']=df['review_score'].apply(lambda x: 1 if x>3 else 0)

            df.drop(['shipping_limit_date','order_purchase_timestamp','order_approved_at','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date','customer_id'], axis=1, inplace=True)

            df['review_availability'] = df['review_comment_message'].apply(lambda x: 1 if x != 'indisponível' else 0)

          
            # data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            # data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            # data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            # data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
           
            # data["review_comment_message"].fillna("No review", inplace=True)

            # data = data.select_dtypes(include=[np.number])
            # cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            # data = data.drop(cols_to_drop, axis=1)

            logging.info("data preprocess strategy is completed")

            return df

        except Exception as e:
            raise e


class DataDivideStrategy(DataStrategy):
    

    def handle_data(self, data: pd.DataFrame) -> Union[np.ndarray, pd.Series]:
        """
        Divides the data into train and test data.
        """
        try:
            X = data.drop("review_score", axis=1)
            y = data["review_score"]

            ohe_payment_type = OneHotEncoder()
            ohe_payment_type.fit(X['payment_type'].values.reshape(-1,1))
            payment_type = ohe_payment_type.transform(X['payment_type'].values.reshape(-1,1)).toarray()
            with open('encoder.pkl', 'wb') as encoder_file:
                pickle.dump(ohe_payment_type, encoder_file)

            strn = StandardScaler()
            strn.fit(X[['price','freight_value',
            'product_height_cm', 'product_width_cm', 'payment_value','purchase_delivery_difference','estimated_actual_delivery_difference']])

            X_strn = strn.transform(X[['price','freight_value',
            'product_height_cm', 'product_width_cm', 'payment_value','purchase_delivery_difference','estimated_actual_delivery_difference']])

            print(f"X_strn shape: {X_strn.shape}")
            
            with open('scaler.pkl', 'wb') as scaler_file:
                pickle.dump(strn, scaler_file)

            X = np.concatenate((X_strn,payment_type,X['review_availability'].values.reshape(-1,1)),axis=1)
            

            imputer = SimpleImputer(strategy='mean')  
            X = imputer.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            print(y_train.shape)
            print(y_train)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(e)
            raise e




class DataCleaning:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        return self.strategy.handle_data(self.df)
