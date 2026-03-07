"""This script loads an MLFLow model from the artifactory and fetches data from the YahooFinance library to classificate
Magnificent 7 stocks: if the price will go up (1) or down (0)"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import dagshub
import mlflow
import numpy as np
import onnxruntime as ort
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from helpers import get_app_features


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


logging.info("Importing env creds")

load_dotenv()

#load DAGSHUB credentials from vault
os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ.get("DAGSHUB_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ.get("DAGSHUB_TOKEN")


logging.info("Accessing DAGSHUB Repository")

repo_url = "https://dagshub.com/erreduarte/mag7-predictor"
mlflow.set_tracking_uri(f"{repo_url}.mlflow")

logging.info("Initizaling MLFlow Client")
client = mlflow.MlflowClient()

MODEL_NAME = "mag_seven_onnxmodel"
MODEL_URI = f"models:/{MODEL_NAME}@production"


models = {}


#before initializing the predict application

@asynccontextmanager # loads model before the app starts
async def lifespan(app: FastAPI):

    # fetches the model directly from DAGSHUB MLFLow artifactory
    try:

        #Get the model path so ONNX open it 
        logging.info("Downloading model")
        model_path = mlflow.artifacts.download_artifacts(artifact_uri=MODEL_URI)

        file_path = os.path.join(model_path, "model.onnx")

        logging.info("Loading downloaded model to the session")
        # models["session"] = ort.InferenceSession(onnx_model.SerializeToString())

        models['session'] = ort.InferenceSession(file_path)

        logging.info("Model session built")

        #before yield: eveything that runs before the app starts
        yield # this is the moment the app is on waiting for inputs (running state)
        #after yield: what runs when the app is turned off
    
    except Exception as e:
        logging.error('Failed to load model: %s', e)
        raise e
    finally:
        models.clear()
        logging.info("Resources cleaned up")


app = FastAPI(lifespan=lifespan) #application initialized



@app.get("/predict/{ticker}")
def predict(ticker:str):

    try:

        logging.info("Predicting outcome")

        session = models.get("session")

        input_data, last_close = get_app_features(str(ticker).upper())

        if input_data.size == 0:
            logging.error("No data to predict")

        # ONNX prediction parameters
        input_name = session.get_inputs()[0].name
        prediction, probabilities = session.run(None, {input_name: input_data})

        prob_dict = probabilities[0]

        max_prob = max(prob_dict.values())

        return {
            "ticker": str(ticker).upper(),
            "reference_data": (datetime.now().date() - timedelta(days=1)).isoformat(),
            "forecast_close_date": datetime.now().date().isoformat(),
            "last_close_value": round(float(last_close),3),
            "label": "UP" if prediction[0] == 1 else "DOWN",
            "probability": f"{round(float(max_prob) * 100,2)}%"

        }
    
    except Exception as e:
        return {"Error predicting": str(e)}
    
    
@app.get('/health')
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9596)
