# AI-Powered Restaurant Visitor Forecasting & Waste Reduction

An end-to-end Machine Learning application designed to forecast daily restaurant visitors, helping restaurant managers optimize food preparation and significantly reduce food waste (overproduction). 

This project combines **Time-Series Forecasting (LSTM)** with **Information Retrieval (FAISS)** to not only predict future demand but also provide Explainable AI (XAI) insights by surfacing similar historical patterns.

## Features
* **Demand Forecasting:** Uses a Deep Learning LSTM neural network to predict daily visitor counts based on historical traffic, rolling averages, and holiday data.
* **Food Waste Risk Analysis:** Automatically compares predicted demand against actual historical data to calculate a percentage-based "Waste Risk" (Overproduction vs. Underproduction).
* **Explainable AI (XAI):** Utilizes FAISS (Cosine Similarity) to retrieve the top 5 most similar past days, providing a human-readable explanation for *why* the AI made its prediction.
* **Interactive Dashboard:** A clean, dependent-dropdown Streamlit UI that visualizes 30-day demand trends and simulated business decisions.
* **Full-Stack Architecture:** Decoupled FastAPI backend and Streamlit frontend.

## Tech Stack
* **Machine Learning:** TensorFlow (Keras), Scikit-Learn, FAISS
* **Data Processing:** Pandas, NumPy
* **Backend API:** FastAPI, Uvicorn
* **Frontend UI:** Streamlit, Matplotlib, Seaborn
* **Language:** Python 3

## Project Structure
```text
Food Demand/
│
├── data/                    # Raw and processed CSV files
├── models/                  # Saved LSTM model (.h5) and Scaler (.pkl)
├── src/                     # Core backend logic
│   ├── preprocessing.py     # Merges Kaggle datasets
│   ├── feature_engineering.py # Creates time-series lags and rolling means
│   ├── train_lstm.py        # Neural network training script
│   ├── retrieval.py         # FAISS vector database logic
│   ├── waste_analysis.py    # Math for waste risk calculations
│   └── api.py               # FastAPI server setup
│
├── dashboard/
│   └── app.py               # Streamlit frontend application
│
├── prediction_log.csv       # Automatically generated logs of all predictions
└── requirements.txt         # Project dependencies
```

## How to Run the Project Locally

**1. Clone the repository and install dependencies:**
```bash
git clone [https://github.com/yourusername/restaurant-demand-ai.git](https://github.com/yourusername/restaurant-demand-ai.git)
cd restaurant-demand-ai
pip install -r requirements.txt
```

**2. Run the Data Pipeline (Only needed once):**
```bash
python src/preprocessing.py
python src/feature_engineering.py
python src/train_lstm.py
```

**3. Start the Backend API:**
Leave this terminal running!
```bash
uvicorn src.api:app --reload
```

**4. Start the Frontend Dashboard:**
Open a *new* terminal window and run:
```bash
streamlit run dashboard/app.py
```
The dashboard will automatically open in your web browser at `http://localhost:8501`.

## How the AI Works (Behind the Scenes)
1. **Feature Engineering:** The system looks at "Lags" (how many people came yesterday, or exactly one week ago) and "Rolling Means" (the 7-day average) to understand momentum.
2. **LSTM Model:** A Long Short-Term Memory neural network evaluates these features alongside holiday indicators to predict foot traffic.
3. **FAISS Retrieval:** The exact same features are fed into a vector database using Inner Product / L2 Normalization (Cosine Similarity). It finds the closest matching historical days to justify the LSTM's prediction, providing a "sanity check" for the user.
