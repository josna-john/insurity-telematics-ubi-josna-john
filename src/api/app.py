from __future__ import annotations
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os, io, json, tempfile
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor

from src.features.featurize import featurize_trip
from src.data.loader import validate_record
from src.gamification.badges import make_badges

APP_MODEL_PATH = os.getenv("MODEL_PATH", "models/gbm_risk.cbm")
APP_FEATS_PATH = os.getenv("FEATNAMES_PATH", "models/gbm_risk_features.json")

app = FastAPI(title="Telematics UBI Scoring API", version="0.2.0")

_model = CatBoostRegressor(); _model.load_model(APP_MODEL_PATH)
with open(APP_FEATS_PATH, "r", encoding="utf-8") as f:
    _feat_names = json.load(f)

class ScoreResponse(BaseModel):
    risk_score: float
    top_contributors: Optional[List[dict]] = None
    badges: Optional[List[dict]] = None

def _assemble_vector(feats: dict, names):
    row = [feats[n] for n in names]
    return np.array(row, dtype=float).reshape(1, -1)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/score/trip", response_model=ScoreResponse)
def score_trip(records: List[dict] = Body(...)):
    for i, rec in enumerate(records, 1):
        ok, msg = validate_record(rec)
        if not ok:
            raise HTTPException(400, f"Invalid record at index {i}: {msg}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
        for rec in records:
            tmp.write(json.dumps(rec) + "\n")
        tmp_path = tmp.name

    feats = featurize_trip(Path(tmp_path))
    X = _assemble_vector(feats, _feat_names)
    score = float(_model.predict(X)[0]); score = max(0.0, min(1.0, score))

    import shap
    explainer = shap.TreeExplainer(_model)
    sv = explainer.shap_values(X)[0]
    pairs = sorted(
        [{"feature": n, "value": float(v)} for n, v in zip(_feat_names, sv)],
        key=lambda d: abs(d["value"]), reverse=True
    )[:8]

    try: os.remove(tmp_path)
    except OSError: pass

    return {"risk_score": score, "top_contributors": pairs, "badges": make_badges(feats)}

@app.post("/score/jsonl", response_model=ScoreResponse)
def score_jsonl(jsonl_text: str = Body(..., media_type="text/plain")):
    records = []
    for i, line in enumerate(io.StringIO(jsonl_text), 1):
        line = line.strip()
        if not line: continue
        try: rec = json.loads(line)
        except Exception as e:
            raise HTTPException(400, f"Bad JSON at line {i}: {e}")
        ok, msg = validate_record(rec)
        if not ok:
            raise HTTPException(400, f"Invalid record at line {i}: {msg}")
        records.append(rec)
    return score_trip(records)
