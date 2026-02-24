import pandas as pd
import numpy as np
import os

# --- PATH SETUP ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_FILE = os.path.join(PROJECT_ROOT,'data', 'test_filtered.csv')
OUTPUT_FILE = os.path.join(PROJECT_ROOT,'data','test_final_dataset.csv')

def process_data():
    print("📢 I AM THE FIXED SCRIPT (VERSION 3.0)")
    print(f"--- Looking for input file at: {INPUT_FILE} ---")
    
    if not os.path.exists(INPUT_FILE):
        print("❌ ERROR: File not found! Run Script 1 first.")
        return

    print("✅ Found file! Loading...")
    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        print("❌ ERROR: Your filtered_vitals.csv is empty!")
        print("Please Run Script 1 again.")
        return

    # 1. Pivot
    print("Pivoting table...")
    df['charttime'] = pd.to_datetime(df['charttime'])
    
    # Pivot and RESET INDEX to keep subject_id as a column
    df_pivot = df.pivot_table(index=['subject_id', 'stay_id', 'charttime'], 
                              columns='itemid', 
                              values='valuenum').reset_index()

    # Rename columns 
    df_pivot = df_pivot.rename(columns={
        220045: 'HeartRate',
        220050: 'SysBP',
        220052: 'MeanBP'
    })

    print(f"DEBUG: Columns before cleaning: {list(df_pivot.columns)}")

    # 2. Clean Missing Values (SAFE METHOD)
    print("Cleaning gaps...")
    df_pivot = df_pivot.sort_values(['subject_id', 'charttime'])
    
    cols_to_fix = ['HeartRate', 'SysBP', 'MeanBP']
    # Only fix columns that actually exist
    existing_cols = [c for c in cols_to_fix if c in df_pivot.columns]
    
    # We update specific columns instead of overwriting the whole dataframe
    if existing_cols:
        df_pivot[existing_cols] = df_pivot.groupby('subject_id')[existing_cols].ffill(limit=30)
    
    # Drop rows that are still empty
    df_pivot = df_pivot.dropna(subset=['MeanBP', 'HeartRate']) 

    # 3. Create Targets
    print("Creating AI Targets...")
    
    # SAFETY CHECK: Ensure subject_id exists before grouping
    if 'subject_id' not in df_pivot.columns:
        print("❌ CRITICAL ERROR: subject_id is missing!")
        print(f"Current columns: {df_pivot.columns}")
        return

    df_pivot['Future_MAP'] = df_pivot.groupby('subject_id')['MeanBP'].shift(-10)
    df_pivot['Hypotension_Next_10min'] = (df_pivot['Future_MAP'] < 65).astype(int)
    
    # Drop rows where we can't see the future
    df_pivot = df_pivot.dropna(subset=['Future_MAP'])

    print(f"Saving {len(df_pivot)} rows to CSV...")
    df_pivot.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ SUCCESS! Final dataset saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_data()