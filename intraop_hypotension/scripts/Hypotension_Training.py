import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
from numpy.lib.stride_tricks import sliding_window_view

# --- CONFIGURATION ---
WINDOW_SIZE = 30
INPUT_FILE = '/content/training_data_complete.csv'
FEATURES = ['HeartRate', 'MeanBP', 'SysBP']
TARGET = 'Hypotension_Next_10min'

def train_safe_model():
    print("--- 🧠 STARTING ROBUST TRAINING (VERSION 2.0) ---")

    # 1. Load Data
    print("Loading Dataset...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    # SAFETY CHECK 1: Remove any rows with missing values
    print(f"Original Row Count: {len(df)}")
    df = df.dropna()
    print(f"Cleaned Row Count:  {len(df)}")

    # 2. Preprocessing
    scaler = StandardScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])

    # 3. Create Windows (Fast NumPy Method)
    print("Creating windows...")
    X_all, y_all = [], []

    for subject_id, group in df.groupby('subject_id'):
        data_val = group[FEATURES].values
        target_val = group[TARGET].values

        if len(data_val) > WINDOW_SIZE:
            # Create sliding windows
            X_subj = sliding_window_view(data_val, window_shape=(WINDOW_SIZE, len(FEATURES)))
            X_subj = X_subj.squeeze(axis=1)
            y_subj = target_val[WINDOW_SIZE:]

            # Trim
            min_len = min(len(X_subj), len(y_subj))
            X_all.append(X_subj[:min_len])
            y_all.append(y_subj[:min_len])

    X = np.concatenate(X_all)
    y = np.concatenate(y_all)
    print(f"✅ Final Training Data Shape: {X.shape}")

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Define Model (Same Architecture)
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=(WINDOW_SIZE, len(FEATURES))),
        MaxPooling1D(2),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # SAFETY CHECK 2: 'clipnorm' prevents the NaN Explosion
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # SAFETY CHECK 3: Slow down if training gets stuck
    lr_schedule = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=1)

    # 6. Train
    print("🚀 Training Started...")
    history = model.fit(X_train, y_train, epochs=10, batch_size=64,
                        validation_data=(X_test, y_test),
                        callbacks=[lr_schedule])

    # 7. Save
    model.save('hypotension_cnn.h5')
    print("✅ Fixed Model Saved as 'hypotension_cnn.h5'")

    # Check if we still have NaNs
    final_loss = history.history['loss'][-1]
    if np.isnan(final_loss):
        print("❌ CRITICAL FAILURE: Loss is still NaN. Data might be corrupted.")
    else:
        print(f"🎉 SUCCESS! Final Loss: {final_loss:.4f} (Should be a real number)")

if __name__ == "__main__":
    train_safe_model()