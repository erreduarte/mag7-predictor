"""This script loads an MLFLow model from the artifactory and fetches data from the YahooFinance library to classificate
Magnificent 7 stocks: if the price will go up (1) or down (0)"""

import boto3
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import onnxruntime as ort
import numpy as np
import logging
from fastapi import FastAPI
from helpers import get_app_features
import uvicorn


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



BUCKET_NAME = "mlflow-artifacts-renan"

MODEL_PATH = "production/model.onnx"
ONNX_FILE = "model.onnx"

models = {}


#before initializing the predict application

@asynccontextmanager # --> load model before the app starts
async def lifespan(app: FastAPI):

    # logging.info("Connecting to MLFLow...") # --> fetches the model directly from MLFLow artifactory
    try:

        logging.info("Initializing AWS client")
        s3 = boto3.client('s3')
              

        logging.info("Downloading %s from %s", ONNX_FILE, f"s3://{MODEL_PATH}")
        s3_model = s3.download_file(BUCKET_NAME, MODEL_PATH, ONNX_FILE)

        logging.info("Loading downloaded model to the session")
        models["session"] = ort.InferenceSession(ONNX_FILE)

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
