import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def create_features(input_path="data/processed_data.csv", output_path="data/final_data.csv", model_dir="models"):
    print("Generating features...")
    df = pd.read_csv(input_path)
    
    df['visit_date'] = pd.to_datetime(df['visit_date'])
    df = df.sort_values(['air_store_id', 'visit_date'])
    
    grouped = df.groupby('air_store_id')
    
    df['lag_1'] = grouped['visitors'].shift(1)
    df['lag_7'] = grouped['visitors'].shift(7)
    df['rolling_mean_7'] = grouped['visitors'].shift(1).rolling(window=7).mean()
    
    df.dropna(inplace=True)

    # --- THE FIX: Save the real unscaled numbers before scaling ---
    df['actual_visitors_unscaled'] = df['visitors']

    features_to_scale = ['visitors', 'lag_1', 'lag_7', 'rolling_mean_7']
    
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    
    df.to_csv(output_path, index=False)
    print(f"Feature engineering complete. Data saved to {output_path}")

if __name__ == "__main__":
    create_features()