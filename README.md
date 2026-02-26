# VITAL-AI — Clinical Decision Support Dashboard

> A production-grade, real-time clinical risk prediction system for surgical ward and ICU monitoring. Two independent ML models trained on real MIMIC-III hospital data.

![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.133-009688) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00) ![XGBoost](https://img.shields.io/badge/XGBoost-3.2-1A6EA0) ![MIMIC-III](https://img.shields.io/badge/data-MIMIC--III-purple)

---

## What It Does

VITAL-AI provides two independent clinical risk predictions via a single unified dashboard:

| Module | Model | Input | Output |
|---|---|---|---|
| **Intraoperative Hypotension** | 1D CNN (Keras) | CSV — HeartRate, MeanBP, SysBP (30-step window) | P(MAP < 65 mmHg within next 10 min) |
| **Postoperative ICU Transfer** | XGBoost Classifier | 12 lab parameters (manual form entry) | P(post-surgery ICU transfer) |

Both models are trained on **MIMIC-III** — the gold-standard real-world ICU clinical dataset from PhysioNet / Beth Israel Deaconess Medical Center.

---

## Risk State System

Both models output a continuous probability (0–100%) mapped to 4 clinical states:

| Score | State | Warning Orb | Voice Alert |
|---|---|---|---|
| 0–39% | 🟢 **STABLE** | Slow green pulse | None |
| 40–59% | 🟡 **CAUTION** | Amber pulse + outer rings | *"Risk elevated at X%. Monitor closely."* |
| 60–79% | 🔴 **DANGER** | Fast red pulse | *"Danger threshold reached. Assess immediately."* |
| 80–100% | 🚨 **CRITICAL BREACH** | Rapid red blink | *"Hypotensive event imminent. Immediate attention required."* |

Voice alerts use the browser's Web Speech API — calm English female voice, auto-triggered on CAUTION+, with a replay button.

---

## Clinical Thresholds

- **Hypotension threshold:** MAP < **65 mmHg** — shown as a dashed reference line on the vitals chart
- **CNN decision boundary:** risk score > 0.5 → DANGER (above hypotension risk)
- **XGBoost decision boundary:** probability ≥ 0.5 → ICU transfer likely

---

## Project Structure

```
miniproj/
├── README.md
├── intraop_hypotension/
│   ├── server.py              # FastAPI backend — serves both models
│   ├── dashboard.html         # Complete frontend (single-file, no build step)
│   ├── requirements.txt       # Python dependencies
│   ├── sample_vitals.csv      # Demo CSV (2 patients at different risk levels)
│   ├── model/
│   │   └── hypotension_cnn.h5 # Trained CNN model weights
│   └── scripts/               # MIMIC-III data processing pipeline
│       ├── 1_filter_data.py   # Filter raw data for vitals itemids
│       ├── 2_process_data.py  # Pivot, ffill, label (MAP < 65 target)
│       └── 3_enrich_data.py   # Merge demographics, outcomes
└── postoperative/
    ├── icu_transfer_xgb.pkl   # Trained XGBoost classifier (~1.8 MB)
    ├── feature_order.pkl      # Exact feature ordering for inference
    └── postoperative.ipynb    # Training + evaluation notebook
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r intraop_hypotension/requirements.txt
```

### 2. Start the server
```bash
# From the miniproj/ root directory
uvicorn intraop_hypotension.server:app --host 0.0.0.0 --port 8000
```

Expected startup output:
```
✅ Intraoperative CNN model loaded!
✅ Postoperative XGBoost model loaded!
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

> **Note:** `oneDNN` TensorFlow warnings on startup are harmless — no GPU required.

### 3. Open the dashboard
Open **http://localhost:8000/** in Chrome or Firefox.

---

## API Reference

### `POST /predict` — Intraoperative
**Request:** multipart/form-data CSV upload

Required CSV columns: `subject_id`, `charttime`, `HeartRate`, `MeanBP`, `SysBP`

**Response:**
```json
{
  "patient_id": "10002",
  "occurrence_time": "2023-01-01 10:25:00",
  "filename_processed": "sample_vitals.csv",
  "risk_score": 0.145,
  "diagnosis": "SAFE (Stable)",
  "vitals": [
    {"step": 1, "HeartRate": 68.0, "MeanBP": 92.0, "SysBP": 125.0}
  ]
}
```

### `POST /predict_post` — Postoperative
**Request:** `application/json`
```json
{
  "age": 60, "gender": 1, "Creatinine": 1.2, "WBC": 8.5,
  "Hemoglobin": 12.0, "Platelet": 200, "Lactate": 1.8,
  "Potassium": 4.2, "Sodium": 138, "diabetes": 0,
  "hypertension": 1, "stroke": 0
}
```

**Response:**
```json
{
  "risk_score": 0.22,
  "diagnosis": "ICU Transfer Unlikely",
  "input_values": { "age": 60, "gender": 1 }
}
```

---

## Using the Dashboard

### Intraoperative Tab
1. Drag & drop (or click to browse) a CSV with columns: `subject_id`, `charttime`, `HeartRate`, `MeanBP`, `SysBP`
2. Click **ANALYZE PATIENT VITALS**
3. The CNN processes the last 30 time steps → risk score, vitals chart, orb state
4. If risk ≥ 40%, a voice alert auto-triggers

Use the included **`sample_vitals.csv`** to test:
- **Patient 10001** — MeanBP declining toward 65 mmHg threshold → higher risk
- **Patient 10002** — Stable vitals throughout → low risk (~14.5%)

### Postoperative Tab
Click **Postoperative** in the sidebar, enter lab values, click **ANALYZE POST-OP RISK**.

**Lab reference ranges (for normal/low-risk values):**

| Parameter | Normal Range | Units |
|---|---|---|
| Age | (actual age) | years |
| Gender | 0 = Male, 1 = Female | — |
| Creatinine | 0.7 – 1.3 | mg/dL |
| WBC | 4.5 – 11.0 | ×10³/µL |
| Hemoglobin | 12 – 17.5 | g/dL |
| Platelet | 150 – 400 | ×10³/µL |
| Lactate | 0.5 – 2.0 | mmol/L |
| Potassium | 3.5 – 5.0 | mEq/L |
| Sodium | 135 – 145 | mEq/L |
| Diabetes | 0 = No, 1 = Yes | — |
| Hypertension | 0 = No, 1 = Yes | — |
| Stroke | 0 = No, 1 = Yes | — |

### Demo Mode (No Backend)
Both tabs have a **"Preview Demo (Offline)"** button that cycles through all 4 risk states with realistic synthetic data — no backend or CSV required. Click it 4 times to see CRITICAL → DANGER → CAUTION → STABLE.

---

## Model Details

### Intraoperative — 1D CNN
- Input shape: `(30, 3)` — 30 time steps × [HeartRate, MeanBP, SysBP]
- Normalization: global means `[85, 75, 115]` ± stds `[20, 15, 20]`
- Padding: zero-padded if patient has < 30 readings
- Label: `Hypotension_Next_10min` — future MAP < 65 mmHg within next 10 min

### Postoperative — XGBoost
- 12 features in fixed order (stored in `feature_order.pkl`)
- Output: `predict_proba()[:,1]` — probability of class 1 (ICU transfer)
- Model size: ~1.8 MB

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| ML — Intraoperative | TensorFlow 2.20 / Keras CNN |
| ML — Postoperative | XGBoost 3.2 |
| Frontend | HTML5 + Tailwind CSS (CDN) + Chart.js |
| Voice Alerts | Web Speech API (browser-native) |
| Dataset | MIMIC-III (PhysioNet) |

---

## Team

Built as a college mini-project demonstrating real-time clinical AI decision support using real hospital data.
