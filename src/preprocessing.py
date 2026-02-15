import pandas as pd
import os

def load_and_clean_data(data_dir="data"):
    """
    Joins air_visit_data, air_store_info, and date_info.
    """
    print("Loading data...")
    visits = pd.read_csv(os.path.join(data_dir, "air_visit_data.csv"))
    store_info = pd.read_csv(os.path.join(data_dir, "air_store_info.csv"))
    date_info = pd.read_csv(os.path.join(data_dir, "date_info.csv"))

    date_info.rename(columns={'calendar_date': 'visit_date'}, inplace=True)

    # visits + store Info
    df = pd.merge(visits, store_info, on="air_store_id", how="left")

    # date Info (holiday, day of week)
    df = pd.merge(df, date_info, on="visit_date", how="left")

    # date to datetime
    df['visit_date'] = pd.to_datetime(df['visit_date'])
    df = df.sort_values(by=['air_store_id', 'visit_date']).reset_index(drop=True)

    
    df['genre_encoded'] = df['air_genre_name'].astype('category').cat.codes
    df['area_encoded'] = df['air_area_name'].astype('category').cat.codes
    

    output_path = os.path.join(data_dir, "processed_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Data merged and saved to {output_path}")
    return df

if __name__ == "__main__":
    load_and_clean_data()