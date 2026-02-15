import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import random

st.set_page_config(page_title="Restaurant AI Dashboard", layout="wide")
st.title("Restaurant Visitor Forecasting & Waste Reduction")
st.markdown("Predict daily visitors to optimize food prep and reduce waste.")

@st.cache_data
def load_data_mapping():
    try:
        df = pd.read_csv("data/final_data.csv")
        mapping = df.groupby('air_store_id')['visit_date'].apply(
            lambda x: sorted(list(set(x)), reverse=True)
        ).to_dict()
        return mapping
    except Exception:
        return {"air_ba937bf13d40fb24": ["2016-05-20"]}

@st.cache_data
def load_store_names():
    try:
        df_info = pd.read_csv("data/air_store_info.csv")
        name_map = {}
        
        prefixes = [
            "Herm's", "Ali's", "Joe's", "Mama", "Papa", "Golden", 
            "Sakura", "Tokyo", "Cozy", "Grand", "Little", "Happy", 
            "Sunset", "Royal", "Urban", "Bella"
        ]
        
        for _, row in df_info.iterrows():
            sid = row['air_store_id']
            genre = str(row['air_genre_name'])
            
            random.seed(sid)
            prefix = random.choice(prefixes)
            
            if "Cafe" in genre or "Sweets" in genre:
                suffix = random.choice(["Bakery", "Cafe", "Coffee Shop", "Deli"])
            elif "Italian" in genre or "French" in genre:
                suffix = random.choice(["Bistro", "Trattoria", "Kitchen", "Deli"])
            elif "Izakaya" in genre or "Dining bar" in genre:
                suffix = random.choice(["Tavern", "Bar & Grill", "Pub", "Lounge"])
            elif "Yakiniku" in genre or "Asian" in genre:
                suffix = random.choice(["BBQ", "Grill", "House"])
            else:
                suffix = random.choice(["Restaurant", "Eatery", "Diner", "Place"])
            
            fake_name = f"{prefix} {suffix}"
            name_map[sid] = f"{fake_name} ({sid[-4:]})"
            
        return name_map
    except Exception:
        return {}

store_date_map = load_data_mapping()
store_list = sorted(list(store_date_map.keys()))
store_names = load_store_names()

st.sidebar.header("Controls")

store_id = st.sidebar.selectbox(
    "Select Restaurant", 
    options=store_list,
    format_func=lambda x: store_names.get(x, x)
)

available_dates = store_date_map.get(store_id, ["2016-05-20"])
date_str = st.sidebar.selectbox("Select Available Date", options=available_dates)

run_btn = st.sidebar.button("Run Forecast")

#  tabs 
tab1, tab2, tab3 = st.tabs(["Forecast & Waste", "Model Evaluation", "Store Insights"])

# global API URL
API_URL = "http://127.0.0.1:8000"

#  tab 1: Forecast & Waste Analysis
with tab1:
    if run_btn:
        try:
            # pred API
            resp = requests.post(f"{API_URL}/predict", json={"store_id": store_id, "date": date_str})
            
            if resp.status_code == 200:
                data = resp.json()
                
                # top level metrics (using the updated column widths)
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1.2])
                col1.metric("Predicted Visitors", data['predicted_visitors'])
                
                #  missing actuals
                if data.get('actual_visitors') is not None:
                    col2.metric("Actual Visitors", data['actual_visitors'], delta=f"{int(data['difference'])} Diff")
                else:
                    col2.metric("Actual Visitors", "N/A (Future)")
                
                # risk logic
                col3.metric("Waste Risk", data['waste_risk'], f"{data['waste_percent']}% Error")
                
                # updated shorter status text
                status = "Optimal" if abs(data['waste_percent']) < 10 else "Review Prep"
                col4.metric("Status", status)
                
                st.divider()

                # explanation & trend
                c1, c2 = st.columns([1, 2])
                
                with c1:
                    st.subheader("AI Explanation")
                    st.info(data['explanation'])
                    
                    st.subheader("Simulated Decision")
                    if data['difference'] > 0:
                        st.warning(f"Over-prediction! You would have wasted ~{int(data['difference'])} meals.")
                    elif data['difference'] < 0:
                        st.error(f"Under-prediction! You lost ~{abs(int(data['difference']))} potential sales.")
                    else:
                        st.success("Perfect prediction! Zero waste.")

                with c2:
                    st.subheader("Demand Trend (Last 30 Days)")
                    hist_df = pd.DataFrame(data['history_30_days'])
                    
                    if not hist_df.empty:
                        fig, ax = plt.subplots(figsize=(10, 4))
                        
                        # plot history 
                        sns.lineplot(data=hist_df, x='visit_date', y='visitors', ax=ax, label='Historical Trend', marker='o')
                        
                        # plot prediction 
                        ax.scatter([data['date']], [data['predicted_visitors']], color='red', s=150, label='AI Forecast', zorder=5)
                        
                        plt.xticks(rotation=45)
                        ax.set_title("Visitor Trend & AI Forecast")
                        ax.legend()
                        st.pyplot(fig)
                    else:
                        st.write("No history available for graph.")

                st.divider()
                
                # similar cases analysis
                st.subheader("Retrieval Analysis (Similar Historical Days)")
                sim_df = pd.DataFrame(data['similar_past_days'])
                
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.dataframe(sim_df, use_container_width=True)
                with sc2:
                    # bar chart 
                    fig2, ax2 = plt.subplots(figsize=(6, 3))
                    dates = [f"Sim {i+1}" for i in range(len(sim_df))]
                    values = sim_df['visitors'].tolist()
                    
                    # qdd prediction
                    dates.append("PREDICTION")
                    values.append(data['predicted_visitors'])
                    colors = ['grey']*len(sim_df) + ['red']
                    
                    ax2.bar(dates, values, color=colors)
                    ax2.set_title("Current Prediction vs Similar Past Days")
                    st.pyplot(fig2)

            else:
                st.error(f"Error: {resp.json().get('detail', 'Unknown Error')}")
        except Exception as e:
            st.error(f"Connection Error: {e}")

# tab2 model eval
with tab2:
    st.header("Model Performance Monitoring")
    
    if st.button("Refresh Evaluation Metrics"):
        try:
            log_df = pd.read_csv("prediction_log.csv")
            if not log_df.empty:
                # metrics
                mae = (log_df['predicted'] - log_df['actual']).abs().mean()
                rmse = ((log_df['predicted'] - log_df['actual']) ** 2).mean() ** 0.5
                accuracy = 100 - (log_df['waste_percent'].abs().mean())
                
                m1, m2, m3 = st.columns(3)
                m1.metric("MAE (Mean Absolute Error)", round(mae, 2))
                m2.metric("RMSE (Root Mean Sq Error)", round(rmse, 2))
                m3.metric("Avg Accuracy %", f"{round(accuracy, 2)}%")
                
                st.subheader("Actual vs Predicted Scatter Plot")
                fig3, ax3 = plt.subplots()
                sns.scatterplot(data=log_df, x='actual', y='predicted', hue='risk_level', ax=ax3)
                ax3.plot([0, log_df['actual'].max()], [0, log_df['actual'].max()], color='red', linestyle='--')
                st.pyplot(fig3)
                
                st.dataframe(log_df)
            else:
                st.warning("Log file exists but is empty.")
            
        except FileNotFoundError:
            st.warning("No prediction log found yet. Run some forecasts in Tab 1 first!")

# tab3: Store Analytics 
with tab3:
    st.header("Store Analytics")
    if st.button("Load Insights"):
        try:
            res = requests.get(f"{API_URL}/store_insights/{store_id}")
            if res.status_code == 200:
                idata = res.json()
                i1, i2, i3 = st.columns(3)
                i1.metric("Average Daily Visitors", idata['avg_visitors'])
                i2.metric("Busiest Day of Week", idata['busiest_day'])
                i3.metric("All-Time Record", f"{idata['max_visitors']} Visitors")
        except:
            st.error("Could not fetch insights. Is the API running?")