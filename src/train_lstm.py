import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import os

def train_model(data_path="data/final_data.csv", model_dir="models"):
    print("Loading data for training...")
    df = pd.read_csv(data_path)
    
    feature_cols = ['lag_1', 'lag_7', 'rolling_mean_7', 'holiday_flg', 'genre_encoded']
    target_col = 'visitors'
    
    X = df[feature_cols].values
    y = df[target_col].values

    # reshape for LSTM [samples, timesteps, features] 
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    
    
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # model
    model = Sequential([
        LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_logarithmic_error')
    
    print("Training...")
    model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_test, y_test), verbose=1)
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save(os.path.join(model_dir, "lstm_model.h5"))
    print("Model saved.")

if __name__ == "__main__":
    train_model()