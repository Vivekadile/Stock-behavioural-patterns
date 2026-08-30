"""AlgoTraders backend - FastAPI.

Design principle (carried over from the whole project's findings, see
services/prediction_service.py docstring): the API exposes a probability
of beating the NIFTY50 median, walk-forward-confirmed historical accuracy,
and a real risk tier - never a fabricated "expected price" or "expected
return %", since those were extensively tested and found to have no
signal beyond the historical average.

Run: uvicorn main:app --reload --port 8000
"""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent))
from routes import companies, predictions, market, health  # noqa: E402

app = FastAPI(
    title="AlgoTraders API",
    description=(
        "NIFTY50 relative-performance probability API. This is NOT investment advice - "
        "see /api/health for status and the project's Project_Report.docx for full methodology "
        "and validated accuracy figures."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to the actual frontend origin before any real deployment
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(companies.router)
app.include_router(predictions.router)
app.include_router(market.router)


@app.get("/")
def root():
    return {
        "name": "AlgoTraders API",
        "docs": "/docs",
        "disclaimer": "Educational/research project. Not investment advice. "
                       "See Project_Report.docx for full validation methodology.",
    }
