# Telematics UBI POC  System Design

## 1) Goals
- Capture raw telematics (speed, accel_long, accel_lat, jerk, heading, gps, time-of-day, road/weather context).
- Clean & aggregate into interpretable features (hard brakes / 100 km, night % miles, speeding exposure).
- Learn driving-behavior embeddings from short windows (530s) via a light sequence encoder.
- Score risk with a calibrated GBM using (aggregates + embeddings)  risk_score in [0, 1].
- Price with GLM/Tweedie using standard rating factors + risk_score for regulator-friendly explainability.
- Expose a FastAPI endpoint and a simple dashboard to show trip feedback + gamified badges.

## 2) Architecture (high-level)
Ingest (stream/samples)  Clean/Validate  Feature/Windowing  
[Behavior Encoder (CNN-LSTM or small Transformer)  embedding]  
GBM Risk Model  Calibration  Pricing (GLM/Tweedie)  API  Dashboard

## 3) Data Schemas

### 3.1 Raw telemetry record (JSONL / CSV)
- timestamp: ISO8601
- speed_mps: float
- accel_long_mps2: float
- accel_lat_mps2: float
- jerk_mps3: float
- heading_deg: float
- gps_lat: float
- gps_lon: float
- time_of_day: {morning, midday, evening, night}
- road_type: {highway, city, rural}
- weather: {clear, rain, snow, fog}
- driver_id, trip_id: strings

### 3.2 Windowed (e.g., 10s windows @ 1020 Hz)
- sequences shaped (T, features), T=100200
- label (if supervised): behavior class (normal/smooth/aggressive) or proxy risk

### 3.3 Aggregates per trip/day
- hard_brake_rate (events / 100 km)
- harsh_accel_rate (events / 100 km)
- cornering_rate (events / 100 km)
- speeding_exposure (% time > posted limit proxy)
- night_mile_share (%)
- rain_mile_share (%)
- avg_speed_p50/p95, std_speed, idle_share, etc.

## 4) Modeling

### 4.1 Behavior Encoder (lightweight)
- Window length: 1030 s; stride: 510 s
- Model: 1D CNN  BiLSTM (or small Transformer with 23 layers)
- Output: 32128D embedding per window; aggregate to trip/day via mean/max-pool

### 4.2 Risk Scorer
- Model: Gradient boosting (CatBoost/XGBoost) with monotonic constraints where sensible
- Inputs: Engineered aggregates + encoder embeddings
- Output: risk_score  [0,1]; calibrated with isotonic or Platt scaling
- Explainability: SHAP for global & per-feature; attention/grad-cam for encoder windows

### 4.3 Pricing (actuarial alignment)
- GLM/Tweedie: Premium = Base * exp(ß? * rating_factors + ? * f(risk_score))
- Keeps pricing transparent for filings while leveraging behavior via risk_score
- Sensitivity curves documented; guardrails (caps/floors, smoothing)

## 5) Real-time Driver Feedback (POC)
- Stream simulator  near-real-time scoring stub (batch in this POC; can be micro-batched)
- Feedback rules: show alerts when thresholds exceeded (hard_brake streaks, speeding exposure)
- Gamification: weekly safety score, streaks, badges

## 6) APIs (FastAPI)
- POST /score/trip  : upload trip (JSON/CSV)  risk_score + top drivers
- GET  /driver/{id}/summary : risk trends, badge status
- GET  /health     : service health

## 7) Evaluation
- Predictive: AUC/PR, Brier score, calibration plots (risk_score vs. proxy labels)
- Operational: latency, throughput (records/sec), memory/CPU footprints
- Business proxy: lift curves vs. GLM-only baseline

## 8) Security & Privacy (POC-aware)
- Pseudonymize driver_id; do not store precise GPS in repo (only samples)
- Config via .env; no secrets in code
- Data retention: small anonymized samples for demo only

## 9) Project Layout
/src
  config.py
  /data        : loaders/validators
  /features    : feature engineering + windowing
  /models      : encoders + GBM + pricing glue
  /api         : FastAPI app
/bin
  simulate_stream.py   # sample generator (JSONL/CSV)
/docs
  pipeline.md, data.md, diagrams (later)
/data
  samples/  # tiny anonymized slices; large data excluded
/models
  saved/ or download pointers

## 10) Run Flow (POC)
1) Generate a sample trip: python .\\bin\\simulate_stream.py --mode aggressive --duration 60 --hz 10
2) (Next step) Clean + featurize: python -m src.features.featurize --input data\\samples\\trip.jsonl
3) (Later) Train encoder/GBM; save models to /models
4) Start API: uvicorn src.api.app:app --reload
5) Score endpoint or upload via dashboard.

