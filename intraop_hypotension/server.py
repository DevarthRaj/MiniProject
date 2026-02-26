from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
import uvicorn
import random
import io
import os
import pickle

app = FastAPI(title="VITAL-AI Clinical Risk API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── INTRAOPERATIVE CNN MODEL ─────────────────────────────────────────────────
CNN_PATH = 'intraop_hypotension/model/hypotension_cnn.h5'
model = None
if os.path.exists(CNN_PATH):
    model = tf.keras.models.load_model(CNN_PATH)
    print("✅ Intraoperative CNN model loaded!")
else:
    print(f"❌ CNN model not found at {CNN_PATH}")

# ── POSTOPERATIVE XGBOOST MODEL ──────────────────────────────────────────────
POST_MODEL_PATH   = 'postoperative/icu_transfer_xgb.pkl'
POST_FEATURE_PATH = 'postoperative/feature_order.pkl'
post_model        = None
post_feature_order = None

if os.path.exists(POST_MODEL_PATH) and os.path.exists(POST_FEATURE_PATH):
    with open(POST_MODEL_PATH,   'rb') as f:
        post_model = pickle.load(f)
    with open(POST_FEATURE_PATH, 'rb') as f:
        post_feature_order = pickle.load(f)
    print("✅ Postoperative XGBoost model loaded!")
    print(f"   Feature order: {post_feature_order}")
else:
    print("❌ Postoperative model files not found")


# ── ROUTES ───────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_dashboard():
    return FileResponse("intraop_hypotension/dashboard.html", media_type="text/html")


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


# ── INTRAOPERATIVE: CSV upload → CNN → risk score + vitals ───────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="CNN model not loaded.")

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    required_cols = ['HeartRate', 'MeanBP', 'SysBP']
    if not all(col in df.columns for col in required_cols):
        raise HTTPException(status_code=400, detail=f"Missing required columns: {required_cols}")
    if 'subject_id' not in df.columns:
        raise HTTPException(status_code=400, detail="Missing required column: subject_id")

    all_patients = df['subject_id'].unique().tolist()
    selected_id  = random.choice(all_patients)
    patient_df   = df[df['subject_id'] == selected_id]

    recent_data  = patient_df[required_cols].tail(30)
    raw_data     = recent_data.values.astype(float)

    global_means = np.array([85.0, 75.0, 115.0])
    global_stds  = np.array([20.0, 15.0, 20.0])
    scaled_data  = (raw_data - global_means) / global_stds

    if len(raw_data) < 30:
        pad_length  = 30 - len(raw_data)
        scaled_data = np.vstack([np.zeros((pad_length, 3)), scaled_data])

    input_tensor = np.expand_dims(scaled_data, axis=0)
    risk_score   = float(model.predict(input_tensor, verbose=0)[0][0])

    vitals = [
        {
            "step":      i + 1,
            "HeartRate": round(float(raw_data[i][0]), 1),
            "MeanBP":    round(float(raw_data[i][1]), 1),
            "SysBP":     round(float(raw_data[i][2]), 1),
        }
        for i in range(len(raw_data))
    ]

    return {
        "patient_id":         str(selected_id),
        "occurrence_time":    str(patient_df.iloc[-1]['charttime']),
        "filename_processed": file.filename,
        "risk_score":         risk_score,
        "diagnosis":          "DANGER (Hypotension)" if risk_score > 0.5 else "SAFE (Stable)",
        "vitals":             vitals,
    }


# ── POSTOPERATIVE: JSON lab values → XGBoost → ICU transfer risk ─────────────
class PostopInput(BaseModel):
    age:          float
    gender:       float
    Creatinine:   float
    WBC:          float
    Hemoglobin:   float
    Platelet:     float
    Lactate:      float
    Potassium:    float
    Sodium:       float
    diabetes:     float
    hypertension: float
    stroke:       float


@app.post("/predict_post")
async def predict_post(data: PostopInput):
    if post_model is None:
        raise HTTPException(status_code=500, detail="Postoperative XGBoost model not loaded.")

    # Assemble features in the exact order the model was trained on
    values = [getattr(data, feat) for feat in post_feature_order]
    X = np.array(values, dtype=float).reshape(1, -1)

    # predict_proba returns [P(no ICU), P(ICU transfer)]
    proba      = post_model.predict_proba(X)[0]
    risk_score = float(proba[1])

    return {
        "risk_score":   risk_score,
        "diagnosis":    "ICU Transfer Likely" if risk_score >= 0.5 else "ICU Transfer Unlikely",
        "input_values": dict(zip(post_feature_order, values)),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)