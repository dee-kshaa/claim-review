"""
app/pipeline/decision_engine.py

Stage 5: Decision Engine
Combines MatchResult + RiskReport into a final typed decision.

Output claim_status values: supported, contradicted, not_enough_information
Severity values: low, medium, high, none, unknown
valid_image: true/false
"""

from __future__ import annotations

import logging

from app.models.schemas import (
    ClaimExtraction,
    ClaimStatus,
    DecisionResult,
    ImageAnalysisReport,
    IssueType,
    MatchResult,
    RiskFlag,
    RiskReport,
    Severity,
)

logger = logging.getLogger(__name__)


def make_decision(
    claim: ClaimExtraction,
    image_report: ImageAnalysisReport,
    match_result: MatchResult,
    risk_report: RiskReport,
) -> DecisionResult:
    """Produce the final claim decision."""

    base_status = match_result.overall_verdict
    final_status = _apply_risk_logic(base_status, match_result, risk_report)

    valid_image = _compute_valid_image(image_report, risk_report)
    severity = _compute_severity(image_report, match_result, final_status, claim)
    issue_type = claim.claimed_issue_type

    # Evidence standard: if contradicted, recompute whether images were actually usable
    esm = match_result.evidence_standard_met
    esm_reason = match_result.evidence_standard_met_reason
    if final_status == ClaimStatus.CONTRADICTED and not esm:
        esm = _recompute_esm_for_contradicted(image_report)

    logger.info(
        "User %s → %s | issue=%s | severity=%s | flags=%s",
        claim.user_id, final_status.value, issue_type.value, severity.value,
        [f.value for f in risk_report.risk_flags],
    )

    return DecisionResult(
        user_id=claim.user_id,
        claim_status=final_status,
        evidence_standard_met=esm,
        evidence_standard_met_reason=esm_reason,
        issue_type=issue_type,
        object_part=claim.claimed_object_part,
        severity=severity,
        supporting_image_ids=match_result.supporting_image_ids,
        risk_flags=risk_report.risk_flags,
        valid_image=valid_image,
    )


def _apply_risk_logic(base: ClaimStatus, match: MatchResult, risk: RiskReport) -> ClaimStatus:
    """Risk can downgrade SUPPORTED; can never upgrade CONTRADICTED."""
    if base != ClaimStatus.SUPPORTED:
        return base

    strong_bad = {RiskFlag.POTENTIAL_MANIPULATION, RiskFlag.NON_ORIGINAL_IMAGE, RiskFlag.WRONG_OBJECT}
    if strong_bad & set(risk.risk_flags):
        max_conf = max(
            (v.confidence for v in match.per_image_verdicts if v.verdict == ClaimStatus.SUPPORTED),
            default=0.0,
        )
        if max_conf < 0.70:
            return ClaimStatus.NOT_ENOUGH_INFORMATION

    return ClaimStatus.SUPPORTED


def _compute_valid_image(image_report: ImageAnalysisReport, risk_report: RiskReport) -> bool:
    bad = {RiskFlag.NON_ORIGINAL_IMAGE, RiskFlag.WRONG_OBJECT}
    if bad & set(risk_report.risk_flags):
        return False
    if not image_report.findings:
        return True
    return not all(not f.is_valid_image for f in image_report.findings)


def _compute_severity(
    image_report: ImageAnalysisReport,
    match_result: MatchResult,
    final_status: ClaimStatus,
    claim: ClaimExtraction,
) -> Severity:
    text = claim.user_claim.lower()

    if final_status == ClaimStatus.NOT_ENOUGH_INFORMATION:
        return Severity.UNKNOWN

    if final_status == ClaimStatus.CONTRADICTED:
        # Check if images show something minor
        for f in image_report.findings:
            if f.detected_damage_present:
                return Severity.LOW
        return Severity.NONE

    # Supported — infer severity from conversation text
    if any(w in text for w in ["shatter", "completely", "badly", "major", "severe", "badly crushed"]):
        return Severity.HIGH
    if any(w in text for w in ["small", "minor", "slight", "tiny", "light"]):
        return Severity.LOW

    # Fallback to image findings severity
    supporting_ids = set(match_result.supporting_image_ids)
    order = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2}
    max_sev = Severity.MEDIUM
    for finding in image_report.findings:
        if finding.image_id in supporting_ids:
            if order.get(finding.severity_estimate, 1) > order.get(max_sev, 1):
                max_sev = finding.severity_estimate
    return max_sev


def _recompute_esm_for_contradicted(image_report: ImageAnalysisReport) -> bool:
    for f in image_report.findings:
        if f.is_valid_image and not f.quality.is_blurry:
            return True
    return False
