# Telematics UBI POC — System Design

This document reflects what’s actually implemented in the repo (CatBoost GBM + monotone constraints; GLM-style pricing; FastAPI + Streamlit).

## 1) Goals
- Ingest raw telematics (speed/accels/jerk/heading/GPS + time-of-day, road, weather).
- Validate schema and types; keep the pipeline **explainable** and **auditable**.
- Engineer interpretable aggregates (event rates, speed/jerk stats, context shares).
- Train a **CatBoost GBM** risk model with **monotonic constraints**.
- Map risk → premium via **GLM-style curve** (caps/floors, slope, pivot).
- Expose a scoring **API (FastAPI)** and a simple **dashboard (Streamlit)** with badges and SHAP contributors.

---

## 2) Architecture (high level)

```bash

Simulator
  ↓
Validation (fast checks)
  ↓
Feature Engineering (aggregates)
  ↓
CatBoost Risk (monotone)
  ↓
SHAP Explainability
  ↓
GLM-style Pricing (caps/floors, slope, pivot)
  ↓
FastAPI (/score/*) + Streamlit UI

```

---

## 3) Data Schemas

### 3.1 Raw telemetry (JSONL)
See `docs/data.md` for the full list and units.

### 3.2 Derived features (per trip)
Produced by `src/features/featurize.py`:
- Trip metadata: `trip_id`, `driver_id`, `n_records`, `hz`, `duration_sec`, `dist_km`
- Events & rates per 100 km: `hard_brakes`, `harsh_accels`, `cornering_events`,
  `hard_brake_rate_100km`, `harsh_accel_rate_100km`, `corner_rate_100km`
- Speed stats: `avg_speed`, `p50_speed`, `p95_speed`, `std_speed`
- Jerk stats: `jerk_mean`, `jerk_p95`
- Context shares: `speeding_exposure`, `idle_share`, `night_mile_share`, `rain_mile_share`

### 3.3 Training table
`data/training/features.csv` contains one row per simulated trip plus:
- `mode` ∈ {smooth, normal, aggressive}
- `target` (proxy risk): smooth=0.1, normal=0.4, aggressive=0.9

---

## 4) Modeling

### 4.1 Risk model (implemented)
- **Algorithm:** CatBoostRegressor (GBM) with **monotonic constraints** on safety features (e.g., more hard braking → never reduces risk).
- **Train/Val split:** 75/25
- **Artifacts:** `models/gbm_risk.cbm` + `models/gbm_risk_features.json`
- **Explainability:** SHAP (global summary + per-trip “top contributors”)
- **Command:**  
  `python -m src.models.train_gbm --data data/training/features.csv --model-out models/gbm_risk.cbm --featnames-out models/gbm_risk_features.json`

### 4.2 Pricing (GLM-style mapping)
- **Formula:** `factor = clamp(exp(intercept + slope * logit(risk)), floor, cap)`
- **Premium:** `premium = base_premium * factor`
- **Controls:** floor (discount cap), cap (surcharge cap), slope (elasticity), pivot (factor≈1 at pivot via intercept).
- **Docs plots:** `docs/pricing/price_curve_*.png`

---

## 5) Real-time scoring (POC)
- `/score/stream` buffers session records in-memory and rescoring on each chunk.
- Useful for demos of rolling risk and the pricing curve; not a production session store.

---

## 6) APIs (FastAPI)
- **`GET /health`** → `{"status":"ok"}`
- **`POST /score/trip`** (JSON array of records) → `risk_score`, `top_contributors`, `badges`
- **`POST /score/jsonl`** (raw JSONL text) → same response
- **`POST /score/stream`** (sessioned) → `session_id`, `n_records`, rolling `risk_score` etc.
- **`DELETE /score/stream/{session_id}`** → clears buffer  
Security: optional `X-API-Key` header (set `API_KEY` env var).

---

## 7) Evaluation
Run:
```bash
python -m src.models.evaluate --data data/training/features.csv --outdir docs/metrics
```

Artifacts:

* `docs/metrics/residuals.png` — residuals vs prediction
* `docs/metrics/score_distribution.png` — validation score histogram
* `docs/metrics/calibration.png` — regression reliability curve
* `docs/metrics/feature_importance.png` (+ `.csv`) — CatBoost gain importance
* `docs/metrics/model_comparison.csv` — leaderboard (CatBoost only in final submission)

---

## 8) Security & Privacy (POC)

* **Simulated data only** in repo; keep real telemetry external and secured.
* Optional API key via `X-API-Key`.
* For production: consented collection, encryption in transit/at rest, role-based access, short retention of raw events.

---

## 9) Project Layout

```
/src
  /api            FastAPI app & auth
  /data           Validation & loaders
  /features       Feature engineering
  /gamification   Badges
  /models         Training, scoring, pricing, explainability
/bin               CLIs (simulate, generate dataset, price, mid-risk helpers)
/data              tiny samples (gitignored except small examples)
/models            model artifacts (.cbm, feature order)
/docs              metrics, explain, pricing, screens
```

---

## 10) Run Flow (end-to-end)

1. **Simulate a trip**
   `python bin/simulate_stream.py --mode aggressive --duration 60 --hz 10 --out data/samples/trip.jsonl`

2. **Featurize (optional one-off)**
   `python -m src.features.featurize --input data/samples/trip.jsonl`

3. **Train model**
   `python -m src.models.train_gbm --data data/training/features.csv --model-out models/gbm_risk.cbm --featnames-out models/gbm_risk_features.json`

4. **Explainability artifacts**
   `python -m src.models.explain --train-data data/training/features.csv --model models/gbm_risk.cbm --featnames models/gbm_risk_features.json --trip data/samples/trip_eval.jsonl --outdir docs/explain`

5. **Run API**
   `uvicorn src.api.app:app --reload` → open `/docs`

6. **Streamlit UI**
   `streamlit run streamlit_app.py` → upload a JSONL, view risk/premium/badges/SHAP.

```
