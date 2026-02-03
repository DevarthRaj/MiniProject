import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numpy.lib.stride_tricks import sliding_window_view

# --- CONFIGURATION ---
INPUT_FILE = '/training_data_complete.csv'
WINDOW_SIZE = 30
FEATURES = ['HeartRate', 'MeanBP', 'SysBP']
TARGET = 'Hypotension_Next_10min'

def run_full_test():
    print("--- 🔄 PREPARING TEST DATA ---")

    # 1. Load & Clean
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"❌ Error: {INPUT_FILE} not found.")
        return
    df = df.dropna()

    # 2. Scale
    scaler = StandardScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])

    # 3. Create Windows
    print("Creating windows...")
    X_all, y_all = [], []
    for subject_id, group in df.groupby('subject_id'):
        data_val = group[FEATURES].values
        target_val = group[TARGET].values
        if len(data_val) > WINDOW_SIZE:
            X_subj = sliding_window_view(data_val, window_shape=(WINDOW_SIZE, len(FEATURES)))
            X_subj = X_subj.squeeze(axis=1)
            y_subj = target_val[WINDOW_SIZE:]
            min_len = min(len(X_subj), len(y_subj))
            X_all.append(X_subj[:min_len])
            y_all.append(y_subj[:min_len])

    if not X_all:
        print("❌ Error: No valid windows created.")
        return

    X = np.concatenate(X_all)
    y = np.concatenate(y_all)

    # 4. Split to get Test Set
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✅ Data Ready! Testing on {len(X_test)} samples.")

    # 5. Run the Test Logic
    print("\n--- 🤖 TESTING FIXED MODEL ---")
    try:
        model = tf.keras.models.load_model('hypotension_cnn.h5')
        print("✅ Model loaded successfully!")
    except:
        print("❌ Error: 'hypotension_cnn.h5' not found.")
        return

    # Pick a random patient
    sample_index = np.random.randint(0, len(X_test))
    input_window = X_test[sample_index]
    true_label = int(y_test[sample_index])

    # Predict
    input_reshaped = np.expand_dims(input_window, axis=0)
    prediction_prob = model.predict(input_reshaped, verbose=0)[0][0]

    # Logic
    predicted_label = 1 if prediction_prob > 0.5 else 0
    pred_text = "DANGER (Hypotension)" if predicted_label == 1 else "SAFE (Stable)"
    true_text = "DANGER" if true_label == 1 else "SAFE"

    # Report
    print(f"\n🔎 CASE REPORT #{sample_index}")
    print("-" * 30)
    print(f"AI Score:     {prediction_prob:.4f}")
    print(f"AI Says:      {pred_text}")
    print(f"Truth Is:     {true_text}")
    print("-" * 30)

    if predicted_label == true_label:
        print("✅ CORRECT PREDICTION")
    else:
        print("❌ INCORRECT PREDICTION")

    # Graph
    plt.figure(figsize=(10,4))
    plt.plot(input_window[:, 1], label="Mean BP (Scaled)", color='red', linewidth=2)
    plt.plot(input_window[:, 0], label="Heart Rate (Scaled)", color='blue', alpha=0.6)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.title(f"AI Input (Last 30 Mins) -> Prediction: {pred_text}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# --- CALL THE FUNCTION TO RUN IT ---
if __name__ == "__main__":
    run_full_test()