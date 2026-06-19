"""
app/pipeline/explanation_generator.py

Stage 6: Explanation Generator
Produces claim_status_justification grounded in claim and image findings.

Output format matches sample_claims.csv — plain text, 1–3 sentences.
No paid API. Uses template + claim context.
"""

from __future__ import annotations

import logging

from app.models.schemas import (
    ClaimExtraction,
    ClaimStatus,
    DecisionResult,
    ImageAnalysisReport,
    IssueType,
    OutputRow,
    RiskFlag,
    Severity,
)

logger = logging.getLogger(__name__)


def generate_explanation(
    claim: ClaimExtraction,
    image_report: ImageAnalysisReport,
    decision_result: DecisionResult,
) -> OutputRow:
    """
    Produce the final output row.

    Args:
        claim:           Parsed claim fields.
        image_report:    Per-image VLM/heuristic findings.
        decision_result: Final decision from Decision Engine.

    Returns:
        OutputRow ready to be written to output.csv.
    """
    justification = _build_justification(claim, image_report, decision_result)

    # Format risk_flags as semicolon-separated, or "none"
    flag_values = [f.value for f in decision_result.risk_flags
                   if f != RiskFlag.NONE]
    risk_flags_str = ";".join(flag_values) if flag_values else "none"

    # Format supporting_image_ids as semicolon-separated, or "none"
    supporting_str = ";".join(decision_result.supporting_image_ids) \
        if decision_result.supporting_image_ids else "none"

    return OutputRow(
        user_id=claim.user_id,
        image_paths=claim.image_paths,
        user_claim=claim.user_claim,
        claim_object=claim.claim_object.value,
        evidence_standard_met="true" if decision_result.evidence_standard_met else "false",
        evidence_standard_met_reason=decision_result.evidence_standard_met_reason,
        risk_flags=risk_flags_str,
        issue_type=decision_result.issue_type.value,
        object_part=decision_result.object_part,
        claim_status=decision_result.claim_status.value,
        claim_status_justification=justification,
        supporting_image_ids=supporting_str,
        valid_image="true" if decision_result.valid_image else "false",
        severity=decision_result.severity.value,
    )


# ---------------------------------------------------------------------------
# Justification builder — template-based, no API needed
# ---------------------------------------------------------------------------

def _build_justification(
    claim: ClaimExtraction,
    image_report: ImageAnalysisReport,
    decision: DecisionResult,
) -> str:
    part = decision.object_part.replace("_", " ")
    obj = decision.issue_type.value.replace("_", " ")
    status = decision.claim_status
    flags = set(decision.risk_flags)

    # Find the best image description
    supporting_ids = set(decision.supporting_image_ids)
    image_desc = _best_image_description(image_report, supporting_ids)

    if status == ClaimStatus.SUPPORTED:
        base = f"The submitted image(s) support the claim by showing {obj} damage on the {part}."
        if image_desc:
            base = f"{image_desc.capitalize()} The claim is supported."
        if RiskFlag.USER_HISTORY_RISK in flags:
            base += " User history also shows some prior claims requiring review."
        if RiskFlag.BLURRY_IMAGE in flags:
            base += " One or more images had quality issues, but at least one image was sufficient."
        return base

    elif status == ClaimStatus.CONTRADICTED:
        base = f"The submitted image(s) do not support the claimed {obj} damage on the {part}."
        if RiskFlag.WRONG_OBJECT in flags:
            base = f"The image does not appear to show the claimed {claim.claim_object.value}."
        elif RiskFlag.CLAIM_MISMATCH in flags and image_desc:
            base = f"{image_desc.capitalize()} This does not match the claimed {obj} on the {part}."
        if RiskFlag.NON_ORIGINAL_IMAGE in flags:
            base += " The image may not be an original photo of the claimed object."
        if RiskFlag.USER_HISTORY_RISK in flags:
            base += " User history also requires review."
        return base

    else:  # NOT_ENOUGH_INFORMATION
        base = (
            f"The submitted image(s) do not provide sufficient evidence to verify "
            f"the claimed {obj} damage on the {part}."
        )
        if RiskFlag.DAMAGE_NOT_VISIBLE in flags:
            base = f"The {part} is not clearly visible in the submitted image(s), so the claim cannot be verified."
        elif RiskFlag.WRONG_ANGLE in flags:
            base = f"The image angle does not show the {part} clearly enough to assess the claimed damage."
        elif RiskFlag.BLURRY_IMAGE in flags:
            base = f"Image quality issues prevent reliable assessment of the claimed {part} damage."
        if RiskFlag.TEXT_INSTRUCTION_PRESENT in flags:
            base += " Any instructions found in the image are disregarded."
        return base


def _best_image_description(report: ImageAnalysisReport, supporting_ids: set) -> str:
    """Get the most informative damage description from supporting images."""
    # Prefer supporting images
    for f in report.findings:
        if f.image_id in supporting_ids and f.detected_damage_present:
            desc = f.detected_damage_description
            if desc and "not available" not in desc.lower() and "failed" not in desc.lower():
                return desc

    # Fallback to any image with damage
    for f in report.findings:
        if f.detected_damage_present:
            desc = f.detected_damage_description
            if desc and "not available" not in desc.lower():
                return desc

    return ""
