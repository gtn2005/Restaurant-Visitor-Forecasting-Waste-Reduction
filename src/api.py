from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predict import VisitorPredictor
from src.retrieval import RetrievalSystem
from src.waste_analysis import calculate_waste_risk
import uvicorn
import pandas as pd
import os
from datetime import datetime

app = FastAPI()

# initialize modules
predictor = VisitorPredictor()
retrieval = RetrievalSystem()

class VisitRequest(BaseModel):
    store_id: str
    date: str

def log_prediction(store_id, date, predicted, actual, risk, waste_pct):
    """Saves prediction results to a CSV for monitoring."""
    log_file = "prediction_log.csv"
    new_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "store_id": store_id,
        "date": date,
        "predicted": predicted,
        "actual": actual,
        "waste_percent": waste_pct,
        "risk_level": risk
    }
    df = pd.DataFrame([new_row])
    if not os.path.exists(log_file):
        df.to_csv(log_file, index=False)
    else:
        df.to_csv(log_file, mode='a', header=False, index=False)

def generate_ai_explanation(predicted, actual, is_holiday, similar_cases):
    """Generates a human-readable explanation."""
    reasons = []
    
    # 1. Holiday Logic
    if is_holiday == 1:
        reasons.append("Today is a holiday, which typically increases traffic.")
    else:
        reasons.append("It is a standard business day (non-holiday).")
    
    # 2. Pattern Logic
    if similar_cases:
        avg_similar = sum([c['visitors'] for c in similar_cases]) / len(similar_cases)
        if abs(predicted - avg_similar) < 5:
            reasons.append(f"The prediction aligns with {len(similar_cases)} similar historical days.")
        elif predicted > avg_similar:
            reasons.append("Demand is trending higher than similar past scenarios.")
        
    # 3. Accuracy Logic
    if actual:
        diff = predicted - actual
        if abs(diff) < 5:
            reasons.append("The model is highly accurate for this scenario.")
        elif diff > 0:
            reasons.append("The model slightly overestimated demand.")
            
    return " ".join(reasons)

@app.post("/predict")
def predict_visitors(request: VisitRequest):
    # 1. Get Features
    features = predictor.get_features(request.store_id, request.date)
    if features is None:
        raise HTTPException(status_code=404, detail="Data not found for this Store/Date combination.")
    
    # 2. Predict & Fetch Actual
    pred_visitors = predictor.predict(features)
    actual_visitors = predictor.get_actual_visitors(request.store_id, request.date)
    history = predictor.get_store_history(request.store_id, request.date)
    
    # 3. Waste Risk (Percentage Based)
    risk, waste_pct, diff = calculate_waste_risk(pred_visitors, actual_visitors)
    
    # 4. Retrieval & Explanation
    similar = retrieval.get_similar_days(features)
    # features[3] is holiday_flg
    explanation = generate_ai_explanation(pred_visitors, actual_visitors, features[3], similar)
    
    # 5. Log it
    log_prediction(request.store_id, request.date, pred_visitors, actual_visitors, risk, waste_pct)
    
    return {
        "store_id": request.store_id,
        "date": request.date,
        "predicted_visitors": pred_visitors,
        "actual_visitors": actual_visitors,
        "difference": diff,
        "waste_percent": waste_pct,
        "waste_risk": risk,
        "explanation": explanation,
        "history_30_days": history,
        "similar_past_days": similar
    }

@app.get("/store_insights/{store_id}")
def get_insights(store_id: str):
    insights = predictor.get_store_insights(store_id)
    if not insights:
        raise HTTPException(status_code=404, detail="Store not found")
    return insights

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)