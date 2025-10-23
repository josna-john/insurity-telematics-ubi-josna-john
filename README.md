# Telematics Integration in Auto Insurance — POC (Josna John)

## What this shows
- **Ingestion → Features → ML Risk Scoring → Pricing → API → Dashboard**
- Interpretable features (hard braking, peak speed, jerk) + SHAP explanations
- GLM-style pricing stub with guardrails (caps/floors), aligned with filings

## Repo map
- `/src` — data loaders, features, models, API
- `/bin` — CLI tools (simulate, dataset gen, price)
- `/models` — saved weights & feature order
- `/docs` — design & explainability plots
- `/data` — small, anonymized samples (real datasets excluded)

## Quickstart
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# generate data + train
python .\bin\generate_dataset.py --trips-per-mode 20
python -m src.models.train_gbm --data .\data\training\features.csv

# run API
uvicorn src.api.app:app --reload
# then open http://127.0.0.1:8000/docs
```

## Score & Explain

```bash
# simulate an eval trip
python .\bin\simulate_stream.py --mode aggressive --duration 60 --hz 10 --out .\data\samples\trip_eval.jsonl

# create SHAP plots
python -m src.models.explain ^
  --train-data .\data\training\features.csv ^
  --model .\models\gbm_risk.cbm ^
  --featnames .\models\gbm_risk_features.json ^
  --trip .\data\samples\trip_eval.jsonl ^
  --outdir .\docs\explain
```

### Artifacts

* `docs/explain/global_shap_summary.png`
* `docs/explain/<trip>_top_shap.png`

## Pricing (stub)

```bash
python .\bin\price_trip.py --trip .\data\samples\trip_eval.jsonl --base-premium 120
```

## Streamlit demo (optional)

```bash
streamlit run .\streamlit_app.py
```

## Notes

* Uses simulated data; public datasets (UAH-DriveSet, AXA) can be added offline.
* See `docs/pipeline.md` for architecture; `src/models/pricing.py` documents guardrails and elasticity.

