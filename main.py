import os
import mlflow
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI


class FetalHealthData(BaseModel):
    accelerations: float
    fetal_movement: float
    uterine_contractions: float
    severe_decelerations: float


app = FastAPI(
    title="Fetal Health API",
    openapi_tags=[
        {"name": "Health", "description": "Get api health"},
        {"name": "Prediction", "description": "Model prediction"},
    ],
)


def load_model():
    """
    Loads a pre-trained model from an MLflow server.

    This function connects to an MLflow server using the provided tracking URI, username,
     and password.
    It retrieves the latest version of the 'fetal_health' model registered on the server.
    The function then loads the model using the specified run ID and returns the loaded model.

    Returns:
        loaded_model: The loaded pre-trained model.

    Raises:
        None
    """
    print("reading model...")
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
    MLFLOW_TRACKING_USERNAME = os.getenv('MLFLOW_TRACKING_USERNAME')
    MLFLOW_TRACKING_PASSWORD = os.getenv('MLFLOW_TRACKING_PASSWORD')
    os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD
    print("setting mlflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print("creating client..")
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    print("getting registered model...")
    registered_model = client.get_registered_model("fetal_health")
    print("read model...")
    run_id = registered_model.latest_versions[-1].run_id
    logged_model = f"runs:/{run_id}/model"
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    print(loaded_model)
    return loaded_model


@app.on_event(event_type="startup")
def startup_event():
    """
    A function that is called when the application starts up. It loads a model into the
    global variable `loaded_model`.

    Parameters:
        None

    Returns:
        None
    """
    global loaded_model
    loaded_model = load_model()


@app.get(path="/", tags=["Health"])
def api_health():
    """
    A function that represents the health endpoint of the API.

    Returns:
        dict: A dictionary containing the status of the API, with the key "status" and
        the value "healthy".
    """
    return {"status": "healthy"}


@app.post(path="/predict", tags=["Prediction"])
def predict(request: FetalHealthData):
    """
    Predicts the fetal health based on the given request data.

    Args:
        request (FetalHealthData): The request data containing the fetal health parameters.

    Returns:
        dict: A dictionary containing the prediction of the fetal health.

    Raises:
        None
    """
    global loaded_model
    received_data = np.array(
        [
            request.accelerations,
            request.fetal_movement,
            request.uterine_contractions,
            request.severe_decelerations,
        ]
    ).reshape(1, -1)
    print(received_data)
    prediction = loaded_model.predict(received_data)
    print(prediction)
    return {"prediction": str(np.argmax(prediction[0]))}