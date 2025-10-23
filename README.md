# Telematics Integration in Auto Insurance — POC (Josna John)

> **Final-Round Deliverable**  
> End-to-end pipeline: simulator → validation → features → GBM risk → GLM-style pricing → API → explainability → dashboard.

## Highlights (what's innovative)
- **Two-stage, regulator-friendly design**: ML risk score (GBM with monotonic constraints) feeding a **GLM-style pricing stub** (caps/floors, elasticity).
- **Explainability-first**: SHAP plots (global + per-trip) and top contributors returned by the API.
- **Gamification**: instant badges (smooth braking, low speeding exposure, gentle handling, daylight driver).
- **Demo-ready**: FastAPI Swagger + Streamlit UI; small anonymized samples only.

---

## Repository structure
```bash

/src
/data            # loaders & validation
/features        # feature engineering
/models          # training, scoring, explainability, pricing, constraints
/api             # FastAPI service
/gamification      # badges (under src/gamification)
/bin               # CLIs (simulate, dataset, price)
/models            # saved weights & feature order (generated)
/data              # samples & derived outputs (gitignored except tiny samples)
/docs              # design & explainability artifacts

```

---

## Quickstart

### Environment
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Generate training data (simulated) and train GBM

```bash
python .\bin\generate_dataset.py --trips-per-mode 20 --duration 60 --hz 10 --out .\data\training\features.csv
python -m src.models.train_gbm --data .\data\training\features.csv --model-out .\models\gbm_risk.cbm --featnames-out .\models\gbm_risk_features.json
```

### Explainability artifacts

```bash
python -m src.models.explain --train-data .\data\training\features.csv --model .\models\gbm_risk.cbm --featnames .\models\gbm_risk_features.json --trip .\data\samples\trip_eval.jsonl --outdir .\docs\explain
```

Artifacts to include in submission:

* `docs/explain/global_shap_summary.png`
* `docs/explain/<trip>_top_shap.png`

### API (FastAPI) — score trips

```bash
uvicorn src.api.app:app --reload
# open http://127.0.0.1:8000/docs
# POST /score/trip with a JSON array of records or /score/jsonl with raw JSONL text
```

### Dashboard (Streamlit)

```bash
streamlit run .\streamlit_app.py
# Upload a JSONL file from data/samples/, view risk, premium, badges, contributors.
```

### Pricing stub (CLI)

```bash
python .\bin\price_trip.py --trip .\data\samples\trip_eval.jsonl --base-premium 120
```

---

## Modeling notes

* **Risk model**: CatBoost GBM; monotonic constraints applied for safety-driving metrics (hard braking, speeding exposure, jerk, etc.).
* **Pricing**: `premium = base_premium * factor`, where `factor = clamp(exp(intercept + slope*logit(risk)), floor, cap)`.
* **Explainability**: SHAP summary + per-trip contributors; API returns top 8 contributors for transparency.
* **Data**: simulator only in repo (no PII). Public datasets (UAH-DriveSet, AXA) can be added offline for extended experiments.

---

## Evaluation

* **Predictive**: RMSE/R² on held-out simulated set (prints during training).
* **Calibration**: risk score is clamped to [0,1]; optional isotonic/Platt can be added.
* **Operational**: micro-batching friendly; features computed in milliseconds for typical trips.

---

## Screenshots to include in submission

1. Swagger `/docs` with a successful **200** from `/score/trip`.
2. Streamlit **Risk/Premium/Badges** header and **Top contributors** chart.
3. SHAP figures:

   * `docs/explain/global_shap_summary.png`
   * `docs/explain/<trip>_top_shap.png`

---


