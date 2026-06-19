"""
app/pipeline/claim_parser.py

Stage 1: Claim Parser
Reads a row from claims.csv and extracts structured claim fields.

Dataset columns: user_id, image_paths, user_claim, claim_object
- claim_object is already given (car/laptop/package) — no need to infer it.
- We parse user_claim (conversation text) to extract issue_type and object_part.
- Uses keyword-based extraction (no paid API needed for HackerRank).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from app.models.schemas import ClaimExtraction, IssueType, ObjectType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Keyword maps for fast CPU-only extraction
# ---------------------------------------------------------------------------

_ISSUE_TYPE_KEYWORDS: dict[IssueType, list[str]] = {
    IssueType.CRACK: ["crack", "cracked", "cracking", "shatter", "shattered", "fracture"],
    IssueType.DENT: ["dent", "dented", "denting", "deform", "bent", "hail"],
    IssueType.SCRATCH: ["scratch", "scratched", "scrape", "scraped", "mark", "scuff"],
    IssueType.BROKEN_PART: ["broken", "broke", "break", "missing", "detach", "fell off",
                             "snapped", "cracked open", "damaged", "collision", "hit"],
    IssueType.WATER_DAMAGE: ["water", "liquid", "spill", "spilled", "wet", "flood",
                              "coffee", "rain", "moisture", "soaked"],
    IssueType.STAIN: ["stain", "stained", "staining", "mark", "oil stain", "oily"],
    IssueType.CRUSHED_PACKAGING: ["crush", "crushed", "crushing", "compressed", "squashed",
                                   "flatten", "collapsed"],
    IssueType.TORN_PACKAGING: ["torn", "tear", "tore", "rip", "ripped", "open", "seal",
                                "opened", "unsealed", "broken seal"],
}

# Object-part keywords per claim_object
_CAR_PART_KEYWORDS: dict[str, list[str]] = {
    "front_bumper": ["front bumper", "bumper front"],
    "rear_bumper": ["rear bumper", "back bumper", "bumper rear", "rear", "back bumper"],
    "windshield": ["windshield", "front glass", "windscreen", "front window"],
    "headlight": ["headlight", "head light", "front light", "head lamp"],
    "taillight": ["taillight", "tail light", "back light", "rear light", "tail lamp"],
    "side_mirror": ["side mirror", "mirror", "wing mirror"],
    "door": ["door", "door panel"],
    "hood": ["hood", "bonnet"],
    "roof": ["roof"],
    "body_panel": ["body panel", "body", "panel", "side panel"],
}

_LAPTOP_PART_KEYWORDS: dict[str, list[str]] = {
    "screen": ["screen", "display", "lcd", "monitor", "glass"],
    "hinge": ["hinge"],
    "keyboard": ["keyboard", "keys", "keycap", "key"],
    "trackpad": ["trackpad", "touchpad"],
    "body": ["body", "case", "shell", "casing", "outer", "lid"],
    "corner": ["corner"],
    "port": ["port", "usb", "connector", "charging"],
}

_PACKAGE_PART_KEYWORDS: dict[str, list[str]] = {
    "package_corner": ["corner"],
    "seal": ["seal", "tape", "flap", "seam"],
    "package_side": ["side", "surface"],
    "contents": ["contents", "inside", "product", "item", "missing"],
    "label": ["label"],
}


def parse_claim(row: dict) -> ClaimExtraction:
    """
    Parse a single claims.csv row into a structured ClaimExtraction.

    Args:
        row: dict with keys: user_id, image_paths, user_claim, claim_object

    Returns:
        ClaimExtraction with typed, validated fields.
    """
    user_id = str(row.get("user_id", "unknown"))
    image_paths = str(row.get("image_paths", ""))
    user_claim = str(row.get("user_claim", ""))
    claim_object_raw = str(row.get("claim_object", "unknown")).strip().lower()

    # claim_object is always given — parse directly
    try:
        claim_object = ObjectType(claim_object_raw)
    except ValueError:
        claim_object = ObjectType.UNKNOWN

    text = user_claim.lower()

    # Extract issue type
    issue_type = _extract_issue_type(text)

    # Extract object part
    object_part = _extract_object_part(text, claim_object)

    return ClaimExtraction(
        user_id=user_id,
        image_paths=image_paths,
        user_claim=user_claim,
        claim_object=claim_object,
        claimed_issue_type=issue_type,
        claimed_object_part=object_part,
    )


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_issue_type(text: str) -> IssueType:
    """Keyword-match to extract the most likely issue type."""
    for issue_type, keywords in _ISSUE_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return issue_type
    return IssueType.UNKNOWN


def _extract_object_part(text: str, claim_object: ObjectType) -> str:
    """Match object part keywords based on claim_object type."""
    if claim_object == ObjectType.CAR:
        part_map = _CAR_PART_KEYWORDS
    elif claim_object == ObjectType.LAPTOP:
        part_map = _LAPTOP_PART_KEYWORDS
    elif claim_object == ObjectType.PACKAGE:
        part_map = _PACKAGE_PART_KEYWORDS
    else:
        return "unknown"

    for part, keywords in part_map.items():
        for kw in keywords:
            if kw in text:
                return part

    return "unknown"
