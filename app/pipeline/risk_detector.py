"""
app/pipeline/risk_detector.py

Stage 4: Risk Detector
Combines image quality signals and user_history.csv data into risk flags.

Risk flags (semicolon-separated in output) match sample_claims.csv:
  none, wrong_object, claim_mismatch, wrong_angle, damage_not_visible,
  blurry_image, user_history_risk, non_original_image, cropped_or_obstructed,
  text_instruction_present, manual_review_required, potential_manipulation

user_history.csv columns:
  user_id, past_claim_count, accept_claim, manual_review_claim,
  rejected_claim, last_90_days_claim_count, history_flags, history_summary
"""

from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

from app.models.schemas import (
    ClaimExtraction,
    ClaimStatus,
    ImageAnalysisReport,
    MatchResult,
    RiskFlag,
    RiskReport,
)

logger = logging.getLogger(__name__)


def detect_risks(
    claim: ClaimExtraction,
    image_report: ImageAnalysisReport,
    match_result: MatchResult,
    history_df: Optional[pd.DataFrame] = None,
) -> RiskReport:
    """Produce a risk report for the claim."""
    flags: List[RiskFlag] = []
    notes: List[str] = []

    # --- Image-level risks ---
    for finding in image_report.findings:
        q = finding.quality

        if q.is_blurry:
            _add(flags, RiskFlag.BLURRY_IMAGE)
            notes.append(f"Image {finding.image_id} is blurry.")

        if q.is_dark:
            _add(flags, RiskFlag.DARK_IMAGE)

        if q.is_low_resolution:
            _add(flags, RiskFlag.LOW_RESOLUTION)

        if q.possible_manipulation or q.text_instruction_detected:
            _add(flags, RiskFlag.TEXT_INSTRUCTION_PRESENT)
            _add(flags, RiskFlag.POTENTIAL_MANIPULATION)
            notes.append(f"Image {finding.image_id} may contain embedded instructions.")

        if q.is_partial:
            _add(flags, RiskFlag.CROPPED_OR_OBSTRUCTED)

        if not finding.is_valid_image:
            _add(flags, RiskFlag.NON_ORIGINAL_IMAGE)

        # Wrong object
        if finding.detected_object_type.value not in ("unknown", claim.claim_object.value):
            _add(flags, RiskFlag.WRONG_OBJECT)

        # Damage region visible but wrong angle / not visible
        if not finding.detected_damage_present and finding.detected_object_type == claim.claim_object:
            _add(flags, RiskFlag.DAMAGE_NOT_VISIBLE)

    # --- Match-level risk: claimed part not found ---
    if match_result.overall_verdict == ClaimStatus.CONTRADICTED:
        _add(flags, RiskFlag.CLAIM_MISMATCH)

    # --- Prompt injection / instruction text in conversation ---
    claim_lower = claim.user_claim.lower()
    injection_phrases = [
        "approve", "skip", "ignore", "bypass", "override", "mark this",
        "accept this", "immediately", "follow it", "follow the note",
        "previous instructions", "system:", "escalate publicly",
        "keep reopening", "mark supported", "mark as supported",
    ]
    if any(phrase in claim_lower for phrase in injection_phrases):
        _add(flags, RiskFlag.TEXT_INSTRUCTION_PRESENT)
        notes.append("Claim conversation contains instruction-like language.")

    # --- User history risk ---
    suspicious_history = False
    if history_df is not None and not history_df.empty:
        suspicious_history, hist_note = _analyse_history(claim.user_id, history_df)
        if suspicious_history:
            _add(flags, RiskFlag.USER_HISTORY_RISK)
            if hist_note:
                notes.append(hist_note)

    # --- Manual review if multiple risk flags ---
    high_risk_flags = {
        RiskFlag.WRONG_OBJECT, RiskFlag.CLAIM_MISMATCH, RiskFlag.TEXT_INSTRUCTION_PRESENT,
        RiskFlag.USER_HISTORY_RISK, RiskFlag.NON_ORIGINAL_IMAGE, RiskFlag.POTENTIAL_MANIPULATION,
    }
    if len(set(flags) & high_risk_flags) >= 1:
        _add(flags, RiskFlag.MANUAL_REVIEW_REQUIRED)

    risk_score = min(len(flags) * 0.12, 1.0)

    return RiskReport(
        user_id=claim.user_id,
        risk_flags=flags if flags else [RiskFlag.NONE],
        risk_score=round(risk_score, 3),
        suspicious_user_history=suspicious_history,
        risk_notes=" | ".join(notes) if notes else None,
    )


def load_history(path: str) -> pd.DataFrame:
    """
    Load user_history.csv.

    Columns: user_id, past_claim_count, accept_claim, manual_review_claim,
             rejected_claim, last_90_days_claim_count, history_flags, history_summary
    """
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        logger.info("Loaded user history: %d rows from %s", len(df), path)
        return df
    except Exception as e:
        logger.warning("Could not load history at %s: %s", path, e)
        return pd.DataFrame()


def _analyse_history(user_id: str, df: pd.DataFrame) -> tuple[bool, str]:
    """
    Use user_history.csv columns to detect risk.
    history_flags column contains "user_history_risk" or "none".
    """
    user_rows = df[df["user_id"].astype(str) == str(user_id)]
    if user_rows.empty:
        return False, ""

    row = user_rows.iloc[0]

    # Direct flag from CSV
    history_flag = str(row.get("history_flags", "none")).strip().lower()
    if history_flag == "user_history_risk":
        summary = str(row.get("history_summary", ""))
        return True, f"User {user_id} flagged in history: {summary}"

    # Heuristic: many rejected claims
    try:
        rejected = int(row.get("rejected_claim", 0))
        past_count = int(row.get("past_claim_count", 0))
        recent = int(row.get("last_90_days_claim_count", 0))

        if past_count >= 5 and rejected >= 2:
            return True, f"User {user_id} has {rejected} rejected claims out of {past_count} total."
        if recent >= 4:
            return True, f"User {user_id} has {recent} claims in the last 90 days."
    except (ValueError, TypeError):
        pass

    return False, ""


def _add(flags: List[RiskFlag], flag: RiskFlag) -> None:
    if flag not in flags:
        flags.append(flag)
