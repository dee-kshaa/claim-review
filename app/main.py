"""
app/main.py

FastAPI application with REST endpoints for the Evidence Review System.

Endpoints:
  POST /claims/verify        — Verify a single claim (multipart form + images)
  POST /claims/batch         — Trigger batch processing of CSV files
  GET  /claims/{claim_id}    — Retrieve a previously processed result
  GET  /health               — Health check

Design rationale:
- Single-claim endpoint accepts multipart so it can receive images directly.
- Batch endpoint accepts file paths (server-side) for pipeline integration.
- Results are cached in memory (replace with Redis/DB in production).
- All errors return structured JSON with an 'error' field.
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Annotated, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.models.schemas import OutputRow
from app.pipeline.runner import run_pipeline, _process_claim, _error_row, load_history
from config.settings import settings

import pandas as pd

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-Modal Evidence Review System",
    description="AI-powered damage claim verification using vision-language models.",
    version="1.0.0",
)

# In-memory result store — replace with Redis or DB in production
_results: Dict[str, OutputRow] = {}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "vlm_backend": settings.VLM_BACKEND}


# ---------------------------------------------------------------------------
# Single claim verification
# ---------------------------------------------------------------------------

@app.post("/claims/verify", response_model=dict)
async def verify_claim(
    claim_id: Annotated[str, Form()] = None,
    user_id: Annotated[str, Form()] = "",
    conversation_text: Annotated[str, Form()] = "",
    images: List[UploadFile] = File(default=[]),
):
    """
    Verify a single damage claim.

    Form fields:
        claim_id          (optional, auto-generated if omitted)
        user_id           (claimant identifier)
        conversation_text (free-form description or conversation)

    Files:
        images[]          (one or more image files)
    """
    if not claim_id:
        claim_id = str(uuid.uuid4())

    if not conversation_text:
        raise HTTPException(status_code=400, detail="conversation_text is required")

    # Save uploaded images to a temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = []
        for upload in images:
            dest = Path(tmpdir) / upload.filename
            with open(dest, "wb") as f:
                f.write(await upload.read())
            image_paths.append(str(dest))

        # Build a minimal pandas Series to reuse _process_claim
        row = pd.Series({
            "claim_id": claim_id,
            "user_id": user_id,
            "conversation_text": conversation_text,
            "image_filenames": "|".join(Path(p).name for p in image_paths),
        })

        try:
            result = _process_claim(row, pd.DataFrame(), tmpdir)
        except Exception as e:
            logger.error("Claim %s error: %s", claim_id, e)
            result = _error_row(claim_id, str(e))

    _results[claim_id] = result
    return result.model_dump()


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

@app.post("/claims/batch")
def batch_process(
    background_tasks: BackgroundTasks,
    claims_path: str = Form(...),
    history_path: Optional[str] = Form(default=None),
    images_dir: str = Form(...),
    output_path: str = Form(default="output.csv"),
):
    """
    Trigger batch processing of a claims CSV file.
    Processing runs in the background; poll /health or check output_path.
    """
    def _run():
        try:
            results = run_pipeline(claims_path, history_path, images_dir, output_path)
            for r in results:
                _results[r.claim_id] = r
        except Exception as e:
            logger.error("Batch processing failed: %s", e)

    background_tasks.add_task(_run)
    return {"status": "accepted", "output_path": output_path}


# ---------------------------------------------------------------------------
# Result retrieval
# ---------------------------------------------------------------------------

@app.get("/claims/{claim_id}")
def get_result(claim_id: str):
    """Retrieve the result for a previously processed claim."""
    if claim_id not in _results:
        raise HTTPException(status_code=404, detail=f"Claim '{claim_id}' not found.")
    return _results[claim_id].model_dump()


@app.get("/claims")
def list_results(limit: int = 50):
    """List the most recently processed claims."""
    recent = list(_results.values())[-limit:]
    return [r.model_dump() for r in recent]
