
import numpy as np
import json
import logging
import pandas as pd
from zenml import pipeline,step


from steps.ingest_data import ingest_df
from steps.clean_data  import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model


from zenml.config import DockerSettings


from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)

from zenml.integrations.constants import MLFLOW, TENSORFLOW
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output
from .utils import get_data_for_test




# define docker setting with mlflow integration
docker_settings=DockerSettings(required_integrations=[MLFLOW])


@step(enable_cache=False)
def dynamic_importer() -> str:
    """Downloads the latest data from a mock API."""
    data = get_data_for_test()
    return data

class DeploymentTriggerConfig(BaseParameters):
	min_accuracy:float=0

@step
def deployment_trigger(
	accuracy:float,
	config:DeploymentTriggerConfig,
):
	return accuracy>=config.min_accuracy


@step(enable_cache=False)
def dynamic_importer()-> str:
	data=get_data_for_test()
	return data


@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service"""

    service.start(timeout=21)  # should be a NOP if already started
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    print("Input Data Shape:", data.shape)
    prediction = service.predict(data)
    return prediction

@step(enable_cache=False)
def prediction_service_loader( pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",):
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    print(existing_services)
    print(type(existing_services))
    return existing_services[0]



@pipeline(enable_cache=False,settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path:str,
	
	min_accuracy:float=0,
	workers:int=1,
	timeout:int=DEFAULT_SERVICE_START_STOP_TIMEOUT
):

	data=ingest_df(data_path=data_path)
	X_train,X_test,y_train,y_test=clean_df(data)
	model=train_model(X_train,X_test,y_train,y_test)
	r2,rmse=evaluate_model(model,X_test,y_test)
	deployment_dicison= deployment_trigger(r2)
	mlflow_model_deployer_step(
		model=model,
		deploy_decision=deployment_dicison,
		workers=workers,
		timeout=timeout,

	)
	


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    # Link all the steps artifacts together
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    prediction=predictor(service=model_deployment_service, data=batch_data)
    return prediction