"""
app/models/schemas.py

Typed data contracts aligned to the HackerRank challenge dataset schema.

Output CSV columns (exact match required):
  user_id, image_paths, user_claim, claim_object,
  evidence_standard_met, evidence_standard_met_reason,
  risk_flags, issue_type, object_part,
  claim_status, claim_status_justification,
  supporting_image_ids, valid_image, severity
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations — values must match sample_claims.csv exactly
# ---------------------------------------------------------------------------

class ObjectType(str, Enum):
    CAR = "car"
    LAPTOP = "laptop"
    PACKAGE = "package"
    UNKNOWN = "unknown"


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_ENOUGH_INFORMATION = "not_enough_information"


class IssueType(str, Enum):
    SCRATCH = "scratch"
    DENT = "dent"
    CRACK = "crack"
    BROKEN_PART = "broken_part"
    WATER_DAMAGE = "water_damage"
    STAIN = "stain"
    CRUSHED_PACKAGING = "crushed_packaging"
    TORN_PACKAGING = "torn_packaging"
    NONE = "none"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    NONE = "none"
    UNKNOWN = "unknown"


class RiskFlag(str, Enum):
    NONE = "none"
    WRONG_OBJECT = "wrong_object"
    CLAIM_MISMATCH = "claim_mismatch"
    WRONG_ANGLE = "wrong_angle"
    DAMAGE_NOT_VISIBLE = "damage_not_visible"
    BLURRY_IMAGE = "blurry_image"
    DARK_IMAGE = "dark_image"
    LOW_RESOLUTION = "low_resolution"
    PARTIAL_OBJECT = "partial_object"
    USER_HISTORY_RISK = "user_history_risk"
    NON_ORIGINAL_IMAGE = "non_original_image"
    CROPPED_OR_OBSTRUCTED = "cropped_or_obstructed"
    TEXT_INSTRUCTION_PRESENT = "text_instruction_present"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"
    POTENTIAL_MANIPULATION = "potential_manipulation"
    DUPLICATE_IMAGE = "duplicate_image"


# ---------------------------------------------------------------------------
# Stage 1 — Claim Parser output
# ---------------------------------------------------------------------------

class ClaimExtraction(BaseModel):
    """Structured information extracted from the claim conversation."""
    user_id: str
    image_paths: str                  # original semicolon-separated string
    user_claim: str                   # original conversation text
    claim_object: ObjectType
    claimed_issue_type: IssueType = IssueType.UNKNOWN
    claimed_object_part: str = "unknown"
    additional_details: Optional[str] = None


# ---------------------------------------------------------------------------
# Stage 2 — Image Analyzer output (per image)
# ---------------------------------------------------------------------------

class ImageQualityFlags(BaseModel):
    is_blurry: bool = False
    is_dark: bool = False
    is_low_resolution: bool = False
    is_partial: bool = False
    possible_manipulation: bool = False
    text_instruction_detected: bool = False
    quality_notes: Optional[str] = None


class ImageFindings(BaseModel):
    """Analysis result for a single image."""
    image_id: str = Field(description="img_N identifier (e.g. img_1)")
    detected_object_type: ObjectType
    detected_damage_present: bool
    detected_damage_description: str
    detected_damage_region: str = Field(description="Which part of object shows damage")
    severity_estimate: Severity
    is_valid_image: bool = True       # false if clearly not original or wrong object
    quality: ImageQualityFlags
    raw_vlm_response: Optional[str] = Field(default=None)


class ImageAnalysisReport(BaseModel):
    """Aggregated findings across all submitted images."""
    user_id: str
    findings: List[ImageFindings]


# ---------------------------------------------------------------------------
# Stage 3 — Evidence Matcher output
# ---------------------------------------------------------------------------

class PerImageVerdict(BaseModel):
    image_id: str
    verdict: ClaimStatus
    confidence: float = Field(ge=0.0, le=1.0)
    match_notes: str


class MatchResult(BaseModel):
    user_id: str
    overall_verdict: ClaimStatus
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    per_image_verdicts: List[PerImageVerdict]
    supporting_image_ids: List[str]


# ---------------------------------------------------------------------------
# Stage 4 — Risk Detector output
# ---------------------------------------------------------------------------

class RiskReport(BaseModel):
    user_id: str
    risk_flags: List[RiskFlag]
    risk_score: float = Field(ge=0.0, le=1.0)
    suspicious_user_history: bool
    risk_notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Stage 5 — Decision Engine output
# ---------------------------------------------------------------------------

class DecisionResult(BaseModel):
    user_id: str
    claim_status: ClaimStatus
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    issue_type: IssueType
    object_part: str
    severity: Severity
    supporting_image_ids: List[str]
    risk_flags: List[RiskFlag]
    valid_image: bool


# ---------------------------------------------------------------------------
# Stage 6 — Final output row (matches output.csv columns exactly)
# ---------------------------------------------------------------------------

class OutputRow(BaseModel):
    """One row in output.csv — column names match the challenge spec exactly."""
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str
    evidence_standard_met: str        # "true" / "false"
    evidence_standard_met_reason: str
    risk_flags: str                   # semicolon-separated or "none"
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: str         # semicolon-separated or "none"
    valid_image: str                  # "true" / "false"
    severity: str
