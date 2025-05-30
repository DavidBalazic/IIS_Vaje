import os
import joblib
import random
import yaml

import numpy as np
import pandas as pd
import tensorflow as tf
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

# Set PYTHONHASHSEED environment variable for reproducibility
os.environ["PYTHONHASHSEED"] = str(random_state)

# Set seeds for Python, NumPy, and TensorFlow
random.seed(random_state)
np.random.seed(random_state)
tf.random.set_seed(random_state)

# Define the model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

for station in stations:
    print(f"\n--- Processing station {station} ---")

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
    
    # Train the model
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    
    # Invert the scaling
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    y_train = pipeline.named_steps["preprocess"].transformers_[0][1].transform(y_train)
    y_test = pipeline.named_steps["preprocess"].transformers_[0][1].transform(y_test)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Test MAE: {mae}")
    print(f"Test MSE: {mse}")
    print(f"Test RMSE: {np.sqrt(mse)}")
    
    # Train the model on the entire dataset
    X_full, y_full = pipeline.fit_transform(df)
    model = build_model((X_full.shape[1], X_full.shape[2]))
    model.fit(X_full, y_full, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    
    # Invert the scaling
    y_full = y_full.reshape(-1, 1)
    y_full = pipeline.named_steps["preprocess"].transformers_[0][1].transform(y_full)
    X_full = X_full.reshape(X_full.shape[0], X_full.shape[1])
    
    # Evaluate the model on the entire dataset
    y_pred_full = model.predict(X_full)
    mse_full = mean_squared_error(y_full, y_pred_full)
    mae_full = mean_absolute_error(y_full, y_pred_full)
    print(f"Full dataset MAE: {mae_full}")
    print(f"Full dataset MSE: {mse_full}")
    print(f"Full dataset RMSE: {np.sqrt(mse_full)}")
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    model.save(f"models/model_{station}.keras")
    
    # Save the pipeline
    joblib.dump(pipeline, f"models/pipeline_{station}.pkl")