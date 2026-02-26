import pandas as pd
import os

# --- CONFIGURATION ---
# Point this to your NEW MIMIC-III file
INPUT_FILE = r'D:\\Project_Group19\\training_data\\CHARTEVENTS.csv' 
OUTPUT_FILE = r'../data/filtered_mimic3.csv' # We save to a new name

# The specific Item IDs
TARGET_IDS = [
    220045,  # Heart Rate
    220050,  # Systolic BP
    220051,  # Diastolic BP
    220052,  # Mean Arterial Pressure (MAP)
    # MIMIC-III sometimes uses different IDs, but let's try these first.
    # If the output is empty, we will need to add MIMIC-III specific IDs.
    211, 220045, # HR
    51, 442, 455, 6701, 220179, 220050, # SysBP
    456, 52, 6702, 443, 220052, 220181, 225312 # MAP
]

def filter_massive_file():
    print(f"--- Starting Extraction from {INPUT_FILE} ---")
    
    if not os.path.exists(INPUT_FILE):
        print("❌ ERROR: File not found! Check the path.")
        return

    chunk_size = 1000000 
    header_written = False
    total_rows = 0
    
    # We read the file in chunks
    # 'low_memory=False' fixes the DtypeWarning you saw
    for i, chunk in enumerate(pd.read_csv(INPUT_FILE, chunksize=chunk_size, 
                                          encoding='utf-8', on_bad_lines='skip', low_memory=False)):
        
        # 1. Compatibility Fix: Rename 'icustay_id' to 'stay_id'
        if 'icustay_id' in chunk.columns:
            chunk = chunk.rename(columns={'icustay_id': 'stay_id'})
        
        # Filter: Keep only the rows that match our IDs
        filtered_df = chunk[chunk['itemid'].isin(TARGET_IDS)]
        
        if not filtered_df.empty:
            mode = 'w' if not header_written else 'a'
            header = not header_written
            
            # Select columns (Now 'stay_id' definitely exists!)
            cols_to_keep = ['subject_id', 'stay_id', 'charttime', 'itemid', 'valuenum']
            
            # Safety check: ensure all columns exist before saving
            existing_cols = [c for c in cols_to_keep if c in filtered_df.columns]
            
            filtered_df[existing_cols].to_csv(OUTPUT_FILE, mode=mode, header=header, index=False)
            
            header_written = True
            total_rows += len(filtered_df)
            
        if (i+1) % 5 == 0:
            print(f"Processed Chunk {i+1}...")

    print(f"✅ DONE! Extracted {total_rows} rows.")
    print(f"Saved to: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    filter_massive_file()