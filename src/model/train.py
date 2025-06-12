import os
import joblib
import random
import yaml

import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow
import dagshub
import tf2onnx

from lxml import etree as ET
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from preprocess import DatePreprocessor, SlidingWindowTransformer

# Load configuration from YAML file
params = yaml.safe_load(open("params.yaml"))["train"]
    
stations = params["stations"]
test_size = params["test_size"]
window_size = params["window_size"]
target_col = params["target_col"]
random_state = params["random_state"]

# MLFlow init
# dagshub.init(repo_owner='DavidBalazic', repo_name='IIS_Vaje', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/DavidBalazic/IIS_Vaje.mlflow')

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

# Set PYTHONHASHSEED environment variable for reproducibility
os.environ["PYTHONHASHSEED"] = str(random_state)

# Set seeds for Python, NumPy, and TensorFlow
random.seed(random_state)
np.random.seed(random_state)
tf.random.set_seed(random_state)

# Define the model
def build_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape, name="input")
    x = LSTM(50, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(50, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# Start MLflow experiment
mlflow.set_experiment("LSTM_Air_Quality_Forecasting")
for station in stations:
    print(f"\n--- Processing station {station} ---")
    
    with mlflow.start_run(run_name=f"{station}_run"):
        mlflow.set_tag("station", station)
        mlflow.log_param("window_size", window_size)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        # Load the preprocessed data
        df = pd.read_csv(f"data/preprocessed/air/{station}.csv")
        df = df[["Date_to", target_col]]

        # Fill missing timestamps
        date_preprocessor = DatePreprocessor("Date_to")
        df = date_preprocessor.fit_transform(df)
        df = df.drop(columns=["Date_to"], axis=1)

        # Use test_size hours of data for testing and the rest for training
        df_test = df.iloc[-test_size:]
        df_train = df.iloc[:-test_size]

        # Preprocess the numeric features
        numeric_transformer = Pipeline([
            ("fillna", SimpleImputer(strategy="mean")),
            ("normalize", MinMaxScaler())
        ])

        preprocess = ColumnTransformer([
            ("numeric_transformer", numeric_transformer, [target_col]),
        ])

        # Add the sliding window transformer to the pipeline
        sliding_window_transformer = SlidingWindowTransformer(window_size)

        # Create the pipeline
        pipeline = Pipeline([
            ("preprocess", preprocess),
            ("sliding_window_transformer", sliding_window_transformer),
        ])

        # Apply the pipeline to the dataframe
        X_train, y_train = pipeline.fit_transform(df_train)
        X_test, y_test = pipeline.transform(df_test)

        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        # Build and train tensorflow keras model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_model(input_shape)
        
        # Enable MLflow autologging for TensorFlow
        # mlflow.tensorflow.autolog(log_models=False, log_input_examples=False, log_datasets=False)

        # Train the model
        early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
        
        # Train the model
        y_pred = model.predict(X_test)

        # Invert the scaling
        scaler = pipeline.named_steps["preprocess"].transformers_[0][1].named_steps["normalize"]
        y_pred_inverted = scaler.inverse_transform(y_pred.reshape(-1, 1))
        y_test_inverted = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Evaluate the model
        mse = mean_squared_error(y_test_inverted, y_pred_inverted)
        mae = mean_absolute_error(y_test_inverted, y_pred_inverted)
        rmse = np.sqrt(mse)
        print(f"Test MAE: {mae}")
        print(f"Test MSE: {mse}")
        print(f"Test RMSE: {rmse}")

        # Log metrics
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_rmse", rmse)

        # Train the model on the entire dataset
        X_full, y_full = pipeline.fit_transform(df)
        model = build_model((X_full.shape[1], X_full.shape[2]))
        model.fit(X_full, y_full, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
        
        # Train the model on the full dataset
        y_pred_full = model.predict(X_full)

        # Invert the scaling
        scaler = pipeline.named_steps["preprocess"].transformers_[0][1].named_steps["normalize"]
        y_pred_full_inverted = scaler.inverse_transform(y_pred_full.reshape(-1, 1))
        y_full_inverted = scaler.inverse_transform(y_full.reshape(-1, 1))

        # Evaluate the model on the entire dataset
        mse_full = mean_squared_error(y_full_inverted, y_pred_full_inverted)
        mae_full = mean_absolute_error(y_full_inverted, y_pred_full_inverted)
        rmse_full = np.sqrt(mse_full)
        print(f"Full dataset MAE: {mae_full}")
        print(f"Full dataset MSE: {mse_full}")
        print(f"Full dataset RMSE: {rmse_full}")

        # Log full dataset metrics
        mlflow.log_metric("full_mae", mae_full)
        mlflow.log_metric("full_mse", mse_full)
        mlflow.log_metric("full_rmse", rmse_full)

        # Save the model
        os.makedirs("models", exist_ok=True)
        model.save(f"models/model_{station}.keras")
        
        # Convert Keras model to ONNX and log it in MLflow
        input_signature = [tf.TensorSpec([None, window_size, 1], tf.float32, name="input")]
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)
        mlflow.onnx.log_model(onnx_model, artifact_path=f"onnx_model_{station}")

        # Save the pipeline
        joblib.dump(pipeline, f"models/pipeline_{station}.pkl")
        mlflow.log_artifact(f"models/pipeline_{station}.pkl")

        # Register the model in MLflow Model Registry
        onnx_model_uri = f"runs:/{mlflow.active_run().info.run_id}/onnx_model_{station}"
        registered_model_name = f"LSTM_ONNX_Model_{station}"
        mlflow.register_model(model_uri=onnx_model_uri, name=registered_model_name)