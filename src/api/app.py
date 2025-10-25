from __future__ import annotations
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os, io, json, uuid, tempfile
from pathlib import Path
import numpy as np
from fastapi.responses import RedirectResponse, Response
import joblib, os, json
from pathlib import Path

from catboost import CatBoostRegressor
import shap

import os
from fastapi import Depends
from src.api.security import require_api_key

from src.features.featurize import featurize_trip
from src.data.loader import validate_record
from src.gamification.badges import make_badges

"""
Telematics UBI Scoring API.

This FastAPI service exposes endpoints to:
  - Score a single trip from JSON records (`/score/trip`)
  - Score raw JSONL text (`/score/jsonl`)
  - Maintain streaming sessions for incremental scoring (`/score/stream`)
It loads a trained model (CatBoost `.cbm` or sklearn `.joblib/.pkl`) and the
feature ordering used by the model, computes a risk score in [0,1], derives
top SHAP contributors, and emits lightweight gamification badges.

Environment variables:
  MODEL_PATH     (default: models/gbm_risk.cbm)
  FEATNAMES_PATH (default: models/gbm_risk_features.json)
"""

APP_MODEL_PATH = os.getenv("MODEL_PATH", "models/gbm_risk.cbm")
APP_FEATS_PATH = os.getenv("FEATNAMES_PATH", "models/gbm_risk_features.json")

app = FastAPI(title="Telematics UBI Scoring API", version="0.3.0")


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def root():
    """
    Basic service info endpoint for GET/HEAD.

    Returns:
        dict: Service metadata and pointers to docs and health checks.
    """
    return {"service": "insurity-ubi-api", "status": "ok", "docs": "/docs", "health": "/health"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """
    No-op favicon handler to avoid 404s from browser probes.

    Returns:
        fastapi.Response: Empty 204 response.
    """
    return Response(status_code=204)


def load_any_model(path):
    """
    Load a persisted model from disk, inferring framework by file suffix.

    Args:
        path (str | Path): Model file path (.cbm for CatBoost, .joblib/.pkl for sklearn).

    Returns:
        tuple[object, str]: (loaded model instance, model kind {'catboost','sklearn'})

    Raises:
        ValueError: If the model suffix is unrecognized.
    """
    p = Path(path)
    if p.suffix == ".cbm":
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(); m.load_model(str(p)); return m, "catboost"
    elif p.suffix in {".joblib", ".pkl"}:
        m = joblib.load(p); return m, "sklearn"
    else:
        raise ValueError(f"Unknown model type: {p.suffix}")


_model, _model_kind = load_any_model(APP_MODEL_PATH)
_feat_names = json.loads(Path(APP_FEATS_PATH).read_text(encoding="utf-8"))

_explainer = shap.TreeExplainer(_model)


class ScoreResponse(BaseModel):
    """
    Standard response schema for scoring endpoints.

    Attributes:
        risk_score (float): Predicted risk in [0,1].
        top_contributors (List[dict] | None): Top SHAP contributors (feature, value).
        badges (List[dict] | None): Gamification badges derived from trip features.
        session_id (str | None): Streaming session identifier for /score/stream.
        n_records (int | None): Number of records accumulated in a stream session.
    """
    risk_score: float
    top_contributors: Optional[List[dict]] = None
    badges: Optional[List[dict]] = None
    session_id: Optional[str] = None
    n_records: Optional[int] = None


def _assemble_vector(feats: dict, names):
    """
    Build a model-ready 2D numpy array from a feature mapping and ordered names.

    Args:
        feats (dict): Feature dictionary {name: value}.
        names (list[str]): Ordered feature names expected by the model.

    Returns:
        np.ndarray: Array shaped (1, n_features).
    """
    row = [feats[n] for n in names]
    return np.array(row, dtype=float).reshape(1, -1)


@app.get("/health")
def health():
    """
    Liveness/health probe.

    Returns:
        dict: {"status": "ok"}
    """
    return {"status": "ok"}


@app.post("/score/trip", response_model=ScoreResponse, dependencies=[Depends(require_api_key)])
def score_trip(records: List[dict] = Body(...)):
    """
    Score a trip from a JSON array of per-record telematics dicts.

    Steps:
        1) Validate each record schema.
        2) Write records to a temporary JSONL file for reuse with the featurizer.
        3) Compute features, assemble vector, predict risk.
        4) Compute SHAP values and select top contributors.
        5) Return risk, contributors, and badges.

    Request Body:
        records (List[dict]): Telematics records (validated by src.data.loader.validate_record).

    Returns:
        ScoreResponse: risk score, SHAP top contributors, badges.

    Raises:
        HTTPException(400): If any record fails validation.
    """
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

    sv = _explainer.shap_values(X)[0]
    pairs = sorted(
        [{"feature": n, "value": float(v)} for n, v in zip(_feat_names, sv)],
        key=lambda d: abs(d["value"]), reverse=True
    )[:8]

    try: os.remove(tmp_path)
    except OSError: pass

    return {"risk_score": score, "top_contributors": pairs, "badges": make_badges(feats)}


@app.post("/score/jsonl", response_model=ScoreResponse, dependencies=[Depends(require_api_key)])
def score_jsonl(jsonl_text: str = Body(..., media_type="text/plain")):
    """
    Score a trip from raw JSON Lines (JSONL) text.

    Args:
        jsonl_text (str): One JSON object per line, each a telematics record.

    Returns:
        ScoreResponse: risk score, SHAP top contributors, badges.

    Raises:
        HTTPException(400): On malformed JSON or invalid record content.
    """
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


_sessions: Dict[str, List[dict]] = {}


class StreamChunk(BaseModel):
    """
    Payload schema for /score/stream incremental scoring.

    Attributes:
        session_id (str | None): Existing session to append to; generated if None.
        records (List[dict]): New telematics records to add to the session buffer.
    """
    session_id: Optional[str] = None
    records: List[dict]


@app.post("/score/stream", response_model=ScoreResponse, dependencies=[Depends(require_api_key)])
def score_stream(chunk: StreamChunk):
    """
    Append records to a stream session and return an updated score.

    Behavior:
        - Creates a new session if session_id is not provided.
        - Validates incoming records and appends to the session buffer.
        - Re-featurizes the entire session (micro-batch) to compute risk.
        - Returns session_id, buffer size, risk, SHAP top contributors, and badges.

    Args:
        chunk (StreamChunk): Session identifier (optional) and records to append.

    Returns:
        ScoreResponse: Includes session_id and n_records for client tracking.

    Raises:
        HTTPException(400): If any record fails schema validation.
    """
    sid = chunk.session_id or str(uuid.uuid4())
    buf = _sessions.setdefault(sid, [])

    for i, rec in enumerate(chunk.records, 1):
        ok, msg = validate_record(rec)
        if not ok:
            raise HTTPException(400, f"Invalid record at index {i}: {msg}")
        buf.append(rec)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
        for rec in buf:
            tmp.write(json.dumps(rec) + "\n")
        tmp_path = tmp.name

    feats = featurize_trip(Path(tmp_path))
    X = _assemble_vector(feats, _feat_names)
    score = float(_model.predict(X)[0]); score = max(0.0, min(1.0, score))

    sv = _explainer.shap_values(X)[0]
    pairs = sorted(
        [{"feature": n, "value": float(v)} for n, v in zip(_feat_names, sv)],
        key=lambda d: abs(d["value"]), reverse=True
    )[:6]

    try: os.remove(tmp_path)
    except OSError: pass

    return {
        "session_id": sid,
        "n_records": len(buf),
        "risk_score": score,
        "top_contributors": pairs,
        "badges": make_badges(feats),
    }


@app.delete("/score/stream/{session_id}", dependencies=[Depends(require_api_key)])
def reset_stream(session_id: str):
    """
    Clear a streaming session buffer and return a confirmation.

    Args:
        session_id (str): The session identifier to clear.

    Returns:
        dict: {"session_id": <id>, "status": "cleared"}
    """
    _sessions.pop(session_id, None)
    return {"session_id": session_id, "status": "cleared"}
