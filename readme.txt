**Project:** Telematics Integration in Auto Insurance — POC (Josna John)
**Repo:** https://github.com/josna-john/insurity-telematics-ubi-josna-john

### What this delivers (one sentence)

Simulator → validation → features → **CatBoost GBM (monotone)** risk → **GLM-style pricing** (caps/floors, pivot, slope) → **FastAPI** scoring API (+ SHAP reasons) → **Streamlit** demo + **evaluation plots**.

---

## Setup

1. **Environment (Windows PowerShell)**

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. **Generate training data & train**

```
python .\bin\generate_dataset.py --trips-per-mode 20 --duration 60 --hz 10 --out .\data\training\features.csv
python -m src.models.train_gbm --data .\data\training\features.csv --model-out .\models\gbm_risk.cbm --featnames-out .\models\gbm_risk_features.json
```

3. **Explainability artifacts**

```
python -m src.models.explain --train-data .\data\training\features.csv ^
  --model .\models\gbm_risk.cbm --featnames .\models\gbm_risk_features.json ^
  --trip .\data\samples\trip_eval.jsonl --outdir .\docs\explain
```

Artifacts: `docs/explain/global_shap_summary.png`, `docs/explain/trip_eval_top_shap.png`

4. **Run API (FastAPI)**

```
$env:API_KEY="change-me"   # optional
uvicorn src.api.app:app --reload
# Swagger: http://127.0.0.1:8000/docs
# If API_KEY is set, send header: X-API-Key: change-me
Demo Video: `docs/screens/swagger_score_trip.mp4`
```

**Quick test**

```
$lines = Get-Content .\data\samples\trip_eval.jsonl -TotalCount 120 | % { $_ | ConvertFrom-Json }
$body  = $lines | ConvertTo-Json -Depth 6
Invoke-RestMethod -Uri "http://127.0.0.1:8000/score/trip" -Method POST -ContentType "application/json" -Headers @{ "X-API-Key"="change-me" } -Body $body
```

5. **Dashboard (Streamlit)**

```
streamlit run .\streamlit_app.py
# Upload a JSONL from data/samples/ to see Risk, Premium, Badges, SHAP reasons.
Demo Video: `docs/screens/streamlit_demo.mp4`
```

6. **Pricing curves (docs)**

```
python -m src.models.price_curve --base-premium 120 --floor 0.75 --cap 1.5 --slope 1.75 --outdir .\docs\pricing
```

---

## How to evaluate (what reviewers can run to evaluate my work)

**A. Model quality**

```
python -m src.models.evaluate --data .\data\training\features.csv --outdir .\docs\metrics
```

Outputs:

* `residuals.png` — residuals vs prediction (should center around 0; patterns = issues)
* `score_distribution.png` — histogram of predicted risk on validation (coverage)
* `calibration.png` — predicted vs actual (closer to diagonal is better)
* `feature_importance.png` — CatBoost gain importances
* `model_comparison.csv` — metrics summary (RMSE, R²)

**B. API scoring**

* Open Swagger at `/docs`, POST to `/score/trip` or `/score/jsonl`.
* Verify response includes: `risk_score` in [0,1] and `top_contributors` (SHAP-style reasons).

**C. Pricing sanity**

* In Streamlit, adjust **floor**, **cap**, **slope**, **pivot** and watch **Premium** update.

  * *Unclamped factor* (caption) shows the raw factor before floor/cap guardrails.
  * Use `data/samples/trip_mid_blend.jsonl` for mid-risk behavior (not clamped).

**D. Streaming feel (optional)**

```
python .\bin\simulate_stream.py --mode aggressive --duration 60 --hz 10 --out .\data\samples\trip_stream.jsonl --trip-id stream01
# Use the 5-chunk PowerShell loop from README.md to call /score/stream repeatedly
```

---

## What the plots mean (plain English)

* **docs/metrics/residuals.png** — If residuals scatter randomly around zero with no funnel shape, errors are stable across the range.
* **docs/metrics/score_distribution.png** — Shows whether the model uses the 0–1 space (not collapsed to a narrow band).
* **docs/metrics/calibration.png** — Binned mean predicted vs actual. Closer to the diagonal means better calibration (risk ~ reality).
* **docs/metrics/feature_importance.png** — Which features the model relies on most (sanity check vs domain intuition).
* **docs/explain/global_shap_summary.png** — Directional impact of features overall: long bars = strong influence; color shows high/low feature value effect.
* **docs/explain/trip_eval_top_shap.png** — Top reasons for this specific trip’s score (customer-facing “why”).
* **docs/pricing/price_curve_factor.png** — Premium factor vs risk; dashed lines = floor/cap guardrails.
* **docs/pricing/price_curve_premium.png** — Dollar premium vs risk for your chosen base premium.

---

## Modeling approach (inputs → outcome)

* **Features:** p95 speed, std speed, hard-brake rate per 100 km, harsh-accel rate, cornering events, jerk stats, night/rain mile share, etc.
* **Model:** **CatBoost GBM** with **monotonic constraints** so risky signals cannot reduce the score; score clipped to [0,1].
* **Pricing:** **GLM-style mapping** `premium = base × clamp(exp(intercept + slope · logit(risk)), floor, cap)`; floor/cap guardrails and a pivot to set where factor ≈ 1.0.
* **Explainability:** SHAP reasons returned by API and displayed in UI.

**Why CatBoost:** ordered boosting → stability/leakage resistance; native monotonic constraints → regulator-friendly; strong accuracy on tabular; fast CPU inference; SHAP-compatible. See `docs/research.md` for citations.

---

## Performance & scalability

* **Runtime:** feature calc ≪ 10 ms per trip; inference ≪ 1 ms on CPU for dozens of features.
* **Service:** stateless **FastAPI**; model loads once; micro-batch via `/score/trip`; streaming via `/score/stream`.
* **Scale:** horizontal scale (multiple workers/instances).
* **Security:** optional API key header `X-API-Key`; HTTPS via host.

---

## Cost & ROI (illustrative)

* **Cost:** CPU-only scoring; small model artifact (`.cbm`); minimal RAM → low cloud run cost.
* **ROI:** usage-based feedback nudges risky miles (speeding/night) down; even small improvements (1–2% claim frequency/severity) materially impact loss ratio. Guardrails stabilize customer bills.

---

## Notes on data & external services

* **Data:** repository uses **simulated** trips only (no PII).
* **Services:** optional deployment configs included for Render (API) and Streamlit Community Cloud (UI).
* **Privacy/Security:** consented telemetry collection recommended; encrypt in transit/at rest; role-based access; short raw retention with aggregated features.

---

## Folder map

```
/src
  /data            loaders & validation
  /features        feature engineering
  /models          training, scoring, explainability, pricing
  /api             FastAPI app + security
  /gamification    safe-driving badges
/bin               CLIs (simulate, dataset, pricing, mid-risk)
/models            trained model + feature order (generated)
/docs              plots (metrics, explainability, pricing)
/data              tiny samples & derived (small files only)
```

**License:** MIT
**Contact:** Josna John ([jojohn@ucsd.edu](mailto:jojohn@ucsd.edu))

---

