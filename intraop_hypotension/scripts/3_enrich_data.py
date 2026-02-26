import pandas as pd
import numpy as np
import os

# --- PATHS (UPDATE THESE IF NEEDED) ---
# Input 1: The processed vitals from Script 2 (Make sure you ran Script 2 first!)
VITALS_FILE = r'D:\\Project_Group19\\data\\test_final_dataset.csv' 

# Input 2: The raw MIMIC-III Context Files
# Check your folder path from the screenshots you sent
ADMISSIONS_FILE = r'D:\\Project_Group19\\training_data\\ADMISSIONS.csv' 
PATIENTS_FILE = r'D:\\Project_Group19\\training_data\\PATIENTS.csv'    

OUTPUT_FILE = r'D:\\Project_Group19\\data\\test_complete.csv'

def enrich_mimic3():
    print("--- 🧬 ENRICHING MIMIC-III DATA ---")
    
    # 1. Load Data
    if not os.path.exists(VITALS_FILE):
        print(f"❌ ERROR: {VITALS_FILE} not found. Did you run Script 2?")
        return

    df_vitals = pd.read_csv(VITALS_FILE)
    df_adm = pd.read_csv(ADMISSIONS_FILE)
    df_pat = pd.read_csv(PATIENTS_FILE)
    
    # 2. Fix Dates (Convert text to Calendar Dates)
    print("Fixing timestamps...")
    df_adm['admittime'] = pd.to_datetime(df_adm['admittime'])
    df_pat['dob'] = pd.to_datetime(df_pat['dob'])

    # 3. Calculate Age (The Missing Column!)
    print("Calculating Patient Ages (Admit Time - DOB)...")
    
    # We merge Admissions and Patients to align the dates
    df_age = pd.merge(df_adm[['subject_id', 'hadm_id', 'admittime']], 
                      df_pat[['subject_id', 'dob', 'gender']], 
                      on='subject_id', how='left')
    
    # Math: Days alive / 365.25 = Years old
    df_age['anchor_age'] = (df_age['admittime'] - df_age['dob']).dt.days // 365.25
    
    # FIX: MIMIC-III sets ages > 89 to 300 for privacy. We cap them at 90.
    df_age.loc[df_age['anchor_age'] > 89, 'anchor_age'] = 90
    
    # Create Gender Flag (Female=1, Male=0)
    df_age['is_female'] = (df_age['gender'] == 'F').astype(int)

    # 4. Prepare Outcomes (Did they die?)
    outcomes = df_adm[['subject_id', 'hospital_expire_flag']]
    # If a patient had multiple visits, check if they died in ANY of them
    death_flags = outcomes.groupby('subject_id')['hospital_expire_flag'].max().reset_index()

    # 5. The Great Merge
    print("Merging context into vitals...")
    
    # We average the age if there are multiple entries (simplest for demo)
    demographics = df_age.groupby('subject_id')[['anchor_age', 'is_female']].mean().reset_index()
    
    # Merge Age/Gender
    df_merged = pd.merge(df_vitals, demographics, on='subject_id', how='left')
    # Merge Death Outcome
    df_merged = pd.merge(df_merged, death_flags, on='subject_id', how='left')
    
    # Cleanup
    df_merged['hospital_expire_flag'] = df_merged['hospital_expire_flag'].fillna(0).astype(int)
    df_merged = df_merged.dropna(subset=['anchor_age']) # Drop rows where age calc failed

    # Save
    df_merged.to_csv(OUTPUT_FILE, index=False)
    print("-" * 30)
    print(f"✅ SUCCESS! Test Data Saved: {OUTPUT_FILE}")
    print(f"Rows: {len(df_merged)}")
    print(f"Columns: {list(df_merged.columns)}")

if __name__ == "__main__":
    enrich_mimic3()