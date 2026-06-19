"""
app/pipeline/runner.py

Batch pipeline runner. Reads claims.csv, user_history.csv, evidence_requirements.csv,
runs all six stages for each claim, and writes output.csv.

Dataset columns (claims.csv):
    user_id, image_paths, user_claim, claim_object

Output columns (output.csv) — must match EXACTLY:
    user_id, image_paths, user_claim, claim_object,
    evidence_standard_met, evidence_standard_met_reason,
    risk_flags, issue_type, object_part,
    claim_status, claim_status_justification,
    supporting_image_ids, valid_image, severity

Usage:
    cd evidence_review
    python -m app.pipeline.runner \\
        --claims data/claims/claims.csv \\
        --history data/claims/user_history.csv \\
        --images . \\
        --output output.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import traceback
from pathlib import Path
from typing import List, Optional

import pandas as pd

from app.models.schemas import (
    ClaimStatus, IssueType, ObjectType, OutputRow, RiskFlag, Severity
)
from app.pipeline.claim_parser import parse_claim
from app.pipeline.image_analyzer import analyze_images
from app.pipeline.evidence_matcher import match_evidence
from app.pipeline.risk_detector import detect_risks, load_history
from app.pipeline.decision_engine import make_decision
from app.pipeline.explanation_generator import generate_explanation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Exact output column order — must match challenge schema
OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason",
    "risk_flags", "issue_type", "object_part",
    "claim_status", "claim_status_justification",
    "supporting_image_ids", "valid_image", "severity",
]


def run_pipeline(
    claims_path: str,
    history_path: Optional[str],
    images_dir: str,
    output_path: str,
) -> List[OutputRow]:
    """
    Run the full 6-stage pipeline for all claims.

    Args:
        claims_path:  Path to claims.csv
        history_path: Path to user_history.csv (optional)
        images_dir:   Base directory for image path resolution
        output_path:  Where to write output.csv

    Returns:
        List of OutputRow objects.
    """
    # Load claims
    claims_df = pd.read_csv(claims_path)
    claims_df.columns = [c.strip().lower() for c in claims_df.columns]
    logger.info("Loaded %d claims from %s", len(claims_df), claims_path)

    # Validate required columns
    required = {"user_id", "image_paths", "user_claim", "claim_object"}
    missing_cols = required - set(claims_df.columns)
    if missing_cols:
        raise ValueError(f"claims.csv is missing required columns: {missing_cols}")

    # Load history
    history_df = load_history(history_path) if history_path else pd.DataFrame()

    results: List[OutputRow] = []

    for idx, row in claims_df.iterrows():
        user_id = str(row.get("user_id", f"row_{idx}"))
        try:
            output = _process_claim(row.to_dict(), history_df, images_dir)
        except Exception as e:
            logger.error("Row %d (user_id=%s) failed: %s\n%s",
                         idx, user_id, e, traceback.format_exc())
            output = _error_row(row.to_dict())
        results.append(output)
        logger.info("user_id=%s → %s", user_id, output.claim_status)

    _write_csv(results, output_path)
    logger.info("Done. %d claims processed → %s", len(results), output_path)
    return results


def _process_claim(row: dict, history_df: pd.DataFrame, images_dir: str) -> OutputRow:
    """Process one claim row through all 6 pipeline stages."""

    # Stage 1 — Claim Parser
    claim = parse_claim(row)

    # Stage 2 — Image Analyzer
    image_report = analyze_images(
        user_id=claim.user_id,
        image_paths_str=claim.image_paths,
        expected_object_type=claim.claim_object,
        images_base_dir=images_dir,
    )

    # Stage 3 — Evidence Matcher
    match_result = match_evidence(claim=claim, report=image_report)

    # Stage 4 — Risk Detector
    risk_report = detect_risks(
        claim=claim,
        image_report=image_report,
        match_result=match_result,
        history_df=history_df if not history_df.empty else None,
    )

    # Stage 5 — Decision Engine
    decision_result = make_decision(
        claim=claim,
        image_report=image_report,
        match_result=match_result,
        risk_report=risk_report,
    )

    # Stage 6 — Explanation Generator
    return generate_explanation(
        claim=claim,
        image_report=image_report,
        decision_result=decision_result,
    )


def _error_row(row: dict) -> OutputRow:
    """Safe fallback row when a claim raises an unexpected exception."""
    return OutputRow(
        user_id=str(row.get("user_id", "unknown")),
        image_paths=str(row.get("image_paths", "")),
        user_claim=str(row.get("user_claim", "")),
        claim_object=str(row.get("claim_object", "unknown")),
        evidence_standard_met="false",
        evidence_standard_met_reason="Pipeline error during processing.",
        risk_flags="none",
        issue_type=IssueType.UNKNOWN.value,
        object_part="unknown",
        claim_status=ClaimStatus.NOT_ENOUGH_INFORMATION.value,
        claim_status_justification="An error occurred during claim processing.",
        supporting_image_ids="none",
        valid_image="true",
        severity=Severity.UNKNOWN.value,
    )


def _write_csv(rows: List[OutputRow], path: str) -> None:
    """Write output rows to CSV with exact column ordering."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump())
    logger.info("Output written to %s", path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Modal Evidence Review Pipeline — HackerRank submission"
    )
    parser.add_argument("--claims", required=True,
                        help="Path to claims.csv")
    parser.add_argument("--history", default=None,
                        help="Path to user_history.csv (optional)")
    parser.add_argument("--images", default=".",
                        help="Base directory for image path resolution (default: current dir)")
    parser.add_argument("--output", default="output.csv",
                        help="Output CSV path (default: output.csv)")
    args = parser.parse_args()

    run_pipeline(
        claims_path=args.claims,
        history_path=args.history,
        images_dir=args.images,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
