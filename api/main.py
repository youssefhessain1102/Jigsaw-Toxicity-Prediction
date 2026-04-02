# api/main.py
import sys
import os
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI

# Make sure src/ modules (model, preprocessor, utils) are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.predict import PredictService
from api.router import router


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────
    config = load_config()
    app.state.predict_service = PredictService(config)
    yield
    # ── Shutdown (nothing to clean up for now) ─────────────────────────────


app = FastAPI(
    title="Toxicity Classifier API",
    description="Multi-label toxicity detection using fine-tuned DistilBERT",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1")