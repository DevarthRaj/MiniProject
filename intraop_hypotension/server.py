from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import uvicorn
import random
import io
import os

app = FastAPI(title="ICU CNN Model API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
MODEL_PATH = 'intraop_hypotension/model/hypotension_cnn.h5'

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
else:
    print(f"❌ ERROR: Model file not found at {MODEL_PATH}")

    model = None


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    required_cols = ['HeartRate', 'MeanBP', 'SysBP']
    all_patients = df['subject_id'].unique().tolist()
    selected_id = random.choice(all_patients)
    patient_df = df[df['subject_id'] == selected_id]
    if not all(col in df.columns for col in required_cols):
        raise HTTPException(status_code=400, detail=f"Missing columns: {required_cols}")
    recent_data = patient_df[required_cols].tail(30)
    raw_data = recent_data.values 
    global_means = np.array([85.0, 75.0, 115.0])
    global_stds = np.array([20.0, 15.0, 20.0])   
    scaled_data = (raw_data - global_means) / global_stds
    if len(recent_data) < 30:
        pad_length = 30 - len(recent_data)
        padding = np.zeros((pad_length, 3)) 
        scaled_data = np.vstack([padding, scaled_data])
        
    input_tensor = np.expand_dims(scaled_data, axis=0)
    risk_score = float(model.predict(input_tensor, verbose=0)[0][0])

    return {
        "patient_id": str(selected_id),
        "occurrence_time": str(patient_df.iloc[-1]['charttime']),
        "filename_processed": file.filename,
        "risk_score": risk_score,
        "diagnosis": "DANGER (Hypotension)" if risk_score > 0.5 else "SAFE (Stable)"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)