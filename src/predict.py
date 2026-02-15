import pandas as pd
import tensorflow as tf
import joblib
import os

class VisitorPredictor:
    def __init__(self, model_path="models/lstm_model.h5", scaler_path="models/scaler.pkl", data_path="data/final_data.csv"):
        # load files
        if not os.path.exists(model_path): 
            raise FileNotFoundError("Model not found")
        
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.data = pd.read_csv(data_path)
        
        # fix date format
        self.data['visit_date'] = pd.to_datetime(self.data['visit_date'])
        
        # input features
        self.feature_cols = ['lag_1', 'lag_7', 'rolling_mean_7', 'holiday_flg', 'genre_encoded']

    def get_features(self, store_id, date_str):
        # filter row
        row = self.data[
            (self.data['air_store_id'] == store_id) & 
            (self.data['visit_date'] == date_str)
        ]
        
        if row.empty:
            return None
        
        return row[self.feature_cols].values[0]

    def get_actual_visitors(self, store_id, date_str):
        """return real visitors."""
        
        row = self.data[
            (self.data['air_store_id'] == store_id) & 
            (self.data['visit_date'] == date_str)
        ]
        
        if row.empty:
            return None
        
        # original value
        return int(row['actual_visitors_unscaled'].values[0])

    def get_store_history(self, store_id, date_str, days=30):
        """get past data."""
        
        current_date = pd.to_datetime(date_str)
        start_date = current_date - pd.Timedelta(days=days)
        
        # filter history
        history = self.data[
            (self.data['air_store_id'] == store_id) & 
            (self.data['visit_date'] >= start_date) & 
            (self.data['visit_date'] < current_date)
        ].sort_values('visit_date')
        
        # format date
        history['visit_date'] = history['visit_date'].dt.strftime('%Y-%m-%d')
        
        # keep needed columns
        history = history[['visit_date', 'actual_visitors_unscaled']]
        history = history.rename(columns={'actual_visitors_unscaled': 'visitors'})
        
        return history.to_dict(orient='records')

    def get_store_insights(self, store_id):
        """basic stats."""
        
        store_data = self.data[self.data['air_store_id'] == store_id].copy()
        
        if store_data.empty:
            return None
        
        # avg and max
        avg_visitors = store_data['actual_visitors_unscaled'].mean()
        max_visitors = store_data['actual_visitors_unscaled'].max()
        
        # busiest weekday
        busiest_day_idx = store_data.groupby(
            store_data['visit_date'].dt.dayofweek
        )['actual_visitors_unscaled'].mean().idxmax()
        
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        return {
            "avg_visitors": int(avg_visitors),
            "max_visitors": int(max_visitors),
            "busiest_day": days[busiest_day_idx]
        }

    def predict(self, features):
        # reshape input
        input_seq = features.reshape((1, 1, len(features)))
        
        # model output
        scaled_pred = self.model.predict(input_seq, verbose=0)[0][0]
        
        # rescale result
        prediction_unscaled = scaled_pred * self.scaler.scale_[0] + self.scaler.mean_[0]
        
        return max(0, int(prediction_unscaled))
