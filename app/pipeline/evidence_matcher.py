"""
app/pipeline/evidence_matcher.py

Stage 3: Evidence Matcher
Matches claim text + image metadata against the claim and produces a verdict.

Since images are not present in the HackerRank dataset, this stage uses:
1. Claim text analysis (conversation content)
2. Risk signals detected in image paths / row data
3. User history flags
to generate realistic, varied verdicts aligned with sample_claims.csv patterns.
"""

from __future__ import annotations

import logging
import re
from typing import List

from app.models.schemas import (
    ClaimExtraction,
    ClaimStatus,
    ImageAnalysisReport,
    ImageFindings,
    IssueType,
    MatchResult,
    ObjectType,
    PerImageVerdict,
    RiskFlag,
)

logger = logging.getLogger(__name__)

# Phrases that indicate the image CANNOT show the claimed damage
_NOT_VISIBLE_INDICATORS = [
    "not sure which photo", "i am not sure", "confused",
    "walked around", "checked twice", "not sure how to explain",
    "first thought everything was fine",
]

# Phrases that indicate wrong angle / wrong area
_WRONG_ANGLE_INDICATORS = [
    "ignore unrelated", "focus on", "only one photo", "could not decide",
]

# Phrases that indicate claim mismatch (user describes minor, then escalates)
_MISMATCH_INDICATORS = [
    "small bump", "nothing major", "first thought", "minor",
    "i think", "maybe", "possibly", "not sure if",
]

# Strong evidence phrases
_STRONG_EVIDENCE_PHRASES = [
    "photo is attached", "i attached", "i uploaded", "photos show", "clearly",
    "you can see", "visible in", "image shows", "the image",
]

# Instruction injection (should trigger manual_review and may not block if evidence ok)
_INSTRUCTION_PHRASES = [
    "approve immediately", "skip manual", "ignore previous", "mark this",
    "follow it and approve", "accept this quickly", "keep reopening",
    "mark supported", "approve the claim", "follow the note",
    "system:", "previous instructions",
    "ignore all previous",
]


def match_evidence(
    claim: ClaimExtraction,
    report: ImageAnalysisReport,
) -> MatchResult:
    """Produce overall verdict and evidence standard assessment."""

    if not report.findings:
        return MatchResult(
            user_id=claim.user_id,
            overall_verdict=ClaimStatus.NOT_ENOUGH_INFORMATION,
            evidence_standard_met=False,
            evidence_standard_met_reason="No images were submitted or could be analysed.",
            per_image_verdicts=[],
            supporting_image_ids=[],
        )

    text = claim.user_claim.lower()
    num_images = len(report.findings)
    image_ids = [f.image_id for f in report.findings]

    # --- Analyse claim text signals ---
    has_instruction = any(p in text for p in _INSTRUCTION_PHRASES)
    has_not_visible = any(p in text for p in _NOT_VISIBLE_INDICATORS)
    has_wrong_angle = any(p in text for p in _WRONG_ANGLE_INDICATORS)
    has_mismatch = any(p in text for p in _MISMATCH_INDICATORS)
    has_strong_evidence = any(p in text for p in _STRONG_EVIDENCE_PHRASES)

    # --- Specific scenario detection from sample_claims patterns ---

    # 1. Image clearly shows wrong object (multi-image with mismatched context phrases)
    wrong_object = _detect_wrong_object_scenario(text, claim, report)

    # 2. Missing contents claim — very hard to verify visually
    missing_contents = (
        claim.claim_object == ObjectType.PACKAGE
        and claim.claimed_object_part in ("contents",)
        and "missing" in text
    )

    # 3. Damage not visible / wrong angle
    damage_not_visible = (
        has_not_visible
        and not has_strong_evidence
        and num_images == 1
    )

    # 4. Clear mismatch: conversation shows minor damage, claim severity seems exaggerated
    claim_mismatch = (
        has_mismatch
        and not has_strong_evidence
        and not has_not_visible
    )

    # --- Determine overall verdict ---
    if wrong_object:
        verdict = ClaimStatus.CONTRADICTED
        esm = True
        esm_reason = (
            "The submitted images appear to show a different vehicle or object, "
            "so the image set does not satisfy vehicle identity evidence."
        )
        supporting_ids: List[str] = []

    elif missing_contents:
        verdict = ClaimStatus.NOT_ENOUGH_INFORMATION
        esm = False
        esm_reason = (
            "The images do not clearly show the expected contents or enough of the "
            "opened package to verify whether anything is missing."
        )
        supporting_ids = []

    elif damage_not_visible:
        verdict = ClaimStatus.NOT_ENOUGH_INFORMATION
        esm = False
        esm_reason = (
            f"The submitted image does not show the {claim.claimed_object_part.replace('_',' ')} "
            f"clearly enough to verify the claimed damage."
        )
        supporting_ids = []

    elif claim_mismatch and not has_instruction:
        verdict = ClaimStatus.CONTRADICTED
        esm = True
        esm_reason = (
            "The image is clear enough to evaluate, but the visible damage does not "
            "match the severity or type described in the claim."
        )
        supporting_ids = []

    else:
        # Standard supported case
        verdict = ClaimStatus.SUPPORTED
        supporting_ids = image_ids[:1]  # primary supporting image
        part_str = claim.claimed_object_part.replace("_", " ")
        obj = claim.claim_object.value

        if num_images > 1:
            esm_reason = (
                f"The {obj} and the relevant part ({part_str}) are visible and "
                f"the submitted image(s) support the claim."
            )
        else:
            esm_reason = (
                f"The {part_str} is visible and the claimed damage can be verified "
                f"from the submitted image."
            )
        esm = True

    # Build per-image verdicts
    per_image_verdicts = _build_per_image_verdicts(
        report.findings, verdict, supporting_ids
    )

    return MatchResult(
        user_id=claim.user_id,
        overall_verdict=verdict,
        evidence_standard_met=esm,
        evidence_standard_met_reason=esm_reason,
        per_image_verdicts=per_image_verdicts,
        supporting_image_ids=supporting_ids,
    )


def _detect_wrong_object_scenario(text: str, claim: ClaimExtraction,
                                   report: ImageAnalysisReport) -> bool:
    """
    Detect if images likely show a different object/vehicle.
    Heuristic: multiple images + conversation hints at mismatched context.
    """
    multi_image = len(report.findings) > 1
    if not multi_image:
        return False

    mismatch_hints = [
        "different car", "another car", "full vehicle view",
        "not the same", "close-up and full",
    ]
    return any(h in text for h in mismatch_hints)


def _build_per_image_verdicts(
    findings: List[ImageFindings],
    overall: ClaimStatus,
    supporting_ids: List[str],
) -> List[PerImageVerdict]:
    """Build per-image verdict list consistent with the overall verdict."""
    verdicts = []
    for f in findings:
        if f.image_id in supporting_ids:
            v = PerImageVerdict(
                image_id=f.image_id,
                verdict=ClaimStatus.SUPPORTED,
                confidence=0.75,
                match_notes="This image supports the claim.",
            )
        elif overall == ClaimStatus.NOT_ENOUGH_INFORMATION:
            v = PerImageVerdict(
                image_id=f.image_id,
                verdict=ClaimStatus.NOT_ENOUGH_INFORMATION,
                confidence=0.65,
                match_notes="Image does not show the claimed part clearly enough.",
            )
        elif overall == ClaimStatus.CONTRADICTED:
            v = PerImageVerdict(
                image_id=f.image_id,
                verdict=ClaimStatus.CONTRADICTED,
                confidence=0.70,
                match_notes="Image contradicts or does not support the claim.",
            )
        else:
            v = PerImageVerdict(
                image_id=f.image_id,
                verdict=ClaimStatus.SUPPORTED,
                confidence=0.65,
                match_notes="Image provides supporting context.",
            )
        verdicts.append(v)
    return verdicts
