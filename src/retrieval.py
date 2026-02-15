import pandas as pd
import numpy as np
import faiss
import os

class RetrievalSystem:
    def __init__(self, data_path="data/final_data.csv"):
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data not found at {data_path}")
        
        self.df = pd.read_csv(data_path)
        
        # features used for similarity
        self.feature_cols = ['lag_1', 'lag_7', 'rolling_mean_7', 'holiday_flg', 'genre_encoded']
        
        # convert to float32 for faiss
        self.vectors = self.df[self.feature_cols].values.astype('float32').copy(order='C')
        
        # normalize for cosine similarity
        faiss.normalize_L2(self.vectors)
        
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        self.index.add(self.vectors)
        
    def get_similar_days(self, query_vector, k=5):
        """return top k similar days."""
        
        # prepare query
        query_vector = np.array([query_vector]).astype('float32').copy(order='C')
        faiss.normalize_L2(query_vector)
        
        distances, indices = self.index.search(query_vector, k)
        
        similar_rows = self.df.iloc[indices[0]].copy()
        
        return similar_rows[['visit_date', 'air_store_id', 'visitors', 'holiday_flg']].to_dict(orient='records')
