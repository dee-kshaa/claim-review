"""
config/settings.py — HackerRank-compatible configuration.
All paid API keys and GPU backends are disabled by default.
"""

from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # VLM backend — set to "heuristic" for CPU-only HackerRank mode
    VLM_BACKEND: str = "heuristic"

    # Image quality thresholds
    BLUR_THRESHOLD: float = 100.0
    DARK_THRESHOLD: float = 50.0
    MIN_PIXEL_COUNT: int = 90_000

    # Risk thresholds
    MAX_CLAIMS_90_DAYS: int = 5
    HIGH_APPROVAL_RATE_THRESHOLD: float = 0.85
    MAX_OBJECT_TYPE_VARIETY: int = 3

    # Evaluation
    EVALUATION_GROUND_TRUTH: str = "evaluation/ground_truth.csv"


settings = Settings()
