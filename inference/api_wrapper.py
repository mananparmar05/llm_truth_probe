"""
api_wrapper.py
--------------
FastAPI endpoint that serves the HallucinationDetector over HTTP.

Usage (after probe training)
-----------------------------
  uvicorn inference.api_wrapper:app --host 0.0.0.0 --port 8000

Endpoints
---------
  POST /detect          — main detection endpoint
  GET  /health          — health check
  GET  /info            — model/probe metadata
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Lazy import: don't load model until first request ─────────────────
_detector = None

MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B")
PROBE_PATH   = os.getenv("PROBE_PATH", "results/models/probe_layer14_logistic_regression.pkl")
BEST_LAYER   = int(os.getenv("BEST_LAYER", "14"))

# ── FastAPI app ───────────────────────────────────────────────────────

app = FastAPI(
    title="LLM Hallucination Detection API",
    description=(
        "Detects hallucinations in LLM outputs using a probing classifier "
        "on transformer hidden states. No model retraining required."
    ),
    version="1.0.0",
)


# ── Pydantic schemas ─────────────────────────────────────────────────

class DetectRequest(BaseModel):
    question: str
    answer: str


class DetectResponse(BaseModel):
    question: str
    answer: str
    hallucination_probability: float
    verdict: str


# ── Startup ───────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Load model and probe on startup."""
    global _detector
    from inference.realtime_detector import HallucinationDetector
    import torch
    _detector = HallucinationDetector.from_pretrained(
        model_name=MODEL_NAME,
        probe_path=PROBE_PATH,
        best_layer=BEST_LAYER,
        torch_dtype=torch.float16,
        device_map="auto",
    )


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _detector is not None}


@app.get("/info")
async def info():
    return {
        "model_name": MODEL_NAME,
        "probe_path": PROBE_PATH,
        "best_layer": BEST_LAYER,
    }


@app.post("/detect", response_model=DetectResponse)
async def detect(request: DetectRequest):
    if _detector is None:
        raise HTTPException(status_code=503, detail="Detector not loaded yet.")
    result = _detector.detect(request.question, request.answer)
    return DetectResponse(
        question=request.question,
        **result,
    )
