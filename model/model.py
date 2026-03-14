"""This script trains and loads a model to DAGSHUB"""

import logging
import os

import dagshub
import mlflow
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

import settings


cfg = settings.Settings()

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
    )


def fetch_data():

    yf_data = yf.download(cfg.mag_seven, cfg.start_date, cfg.end_date, group_by="ticker")
    yf_data = yf_data.stack(level=0, future_stack = True).reset_index()
    yf_data.columns.name = None


    df = yf_data[["Date", "Ticker", "Close", "High", "Low", "Open", "Volume"]].drop_duplicates()
    
    # padronize column names
    df.columns = df.columns.str.lower()

    # enforce datetime to date column
    df.date = pd.to_datetime(df.date)

    # enforce data types  
    df[['close', 'high', 'low', 'open']] = (
        df[['close', 'high', 'low', 'open']]
        .astype(float)
        )        
    
    df.volume = df.volume.astype(int)

    # sort columns by date asc and ticker
    df = df.sort_values(by=['date', 'ticker'])

    print(df.head())

    logging.info("Dataframe ready")

    return df


def prepare_data(df):

    df['today'] = df.groupby('ticker')['close'].pct_change(1) * 100

    for i in range(1,6):
        df[f'lag_{i}'] = df.groupby('ticker')['today'].shift(i)


    df['volume_billions'] = df['volume'] / 1_000_000_000

    df['ma_10'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=10).mean())
    df['dist_ma10'] = (df['close'] - df['ma_10']) / df['ma_10'].shift(1)

    df['target'] = (df['today'] > 0).astype(int)

    df = df.dropna()

    features_outliers = ['volume_billions', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'dist_ma10']

    # outliers cleaning
    for col in features_outliers:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[(df[col] >= lower_bound) & (df[col] < upper_bound)]

    print(df.head(100))
    return df

    
def features_preparation(df):

    # features selection
    features = ['volume_billions', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'dist_ma10']

    #time split
    train = df.query('date.dt.year in (2023, 2024, 2025)').copy() #train with data from the past (year 2023-2025)
    # train = train[(np.abs(stats.zscore(train[features])) < 4).all(axis=1)]  # outliers 

    validate = df.query('date.dt.year == 2026').copy() #validate with the future (year 2026)
    

    #data to train
    X_train = train[features]
    y_train = train['target']

    #data to validate
    X_val = validate[features]
    y_val = validate['target']


    return X_train, y_train, X_val, y_val


def mlflow_log_model(X_train, y_train, X_val, y_val):   

    #sk-learn logistic regression pipeline

    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            max_iter=1000,
            random_state=42, 
            solver = 'lbfgs', 
            class_weight='balanced'))
            ])
    
    
    #### SK LEARN DETAILS ####
    # fit the data        
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val) # 0 or 1
    y_proba = pipeline.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    ras = roc_auc_score(y_val, y_proba)

    if acc > 0.5 and ras > 0.5:
        logging.info("Success! Model approved with ACC: %.2f and RAS: %.2f", acc, ras)

        mlflow.sklearn.autolog()
        with mlflow.start_run(
            description=f"mag7 training model from {(cfg.start_date)} to {(cfg.end_date)}"):

                #LOG MANUAL METRICS
                mlflow.log_metric("sk-accuracy_score", acc)
                mlflow.log_metric("sk-roc_auc-score", ras)


                # creates the model signature: show MLFLow the input and output formats
                signature = infer_signature(X_train, pipeline.predict(X_train))

                #### ONNX DETAILS ####

                # describe the features to ONNX
                initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]

                # convert the pipeline to ONNX
                onnx_model = convert_sklearn(pipeline, initial_types=initial_type)

                
                #log the SK-LEARN model
                mlflow.sklearn.log_model(
                        pipeline,
                        name="sk-model",
                        signature=signature,
                        registered_model_name= cfg.skmodel_name,
                        # saves the first 5 lines of input for training
                        input_example=X_train.iloc[:5]
                )

                logging.info("SKLearn Model successfully logged")

                #log the ONNX model
                mlflow.onnx.log_model(
                        onnx_model,
                        name="onnx-model",
                        signature=signature,
                        registered_model_name= cfg.onnx_model_name,
                        input_example=X_train.iloc[:5]
                )

                logging.info("ONNX Model successfully logged")

                
    else:
        logging.warning("MODEL NOT APPROVED. Insufficient metrics. ACC: %.2f, RAS: %.2f", acc, ras)
        return None      
    


def main():

    df = fetch_data()
    data = prepare_data(df)
    X_train, y_train, X_val, y_val = features_preparation(data)


    logging.info("Loading DAGHUB credentials")
    load_dotenv()
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ.get("DAGSHUB_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ.get("DAGSHUB_TOKEN")

    logging.info("Initializing DAGSHUB Repository and experiment")
    dagshub.init(repo_owner="erreduarte", repo_name="mag7-predictor", mlflow=True)
    mlflow.set_experiment(cfg.mlflow_experiment_name)

    mlflow_log_model(X_train, y_train, X_val, y_val)     

  

if __name__ == "__main__":
    main()

