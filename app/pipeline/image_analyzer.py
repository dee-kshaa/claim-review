"""
app/pipeline/image_analyzer.py

Stage 2: Image Analyzer
Runs each submitted image through OpenCV quality checks and the VLM backend.

Dataset image_paths format: semicolon-separated relative paths
  e.g. "images/test/case_001/img_1.jpg;images/test/case_001/img_2.jpg"

Since images are NOT provided in the HackerRank dataset (only paths are listed),
this stage gracefully handles missing files and produces sensible defaults.
The image_id is extracted as img_N from the path (e.g. "img_1").
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image

from app.models.schemas import (
    ImageAnalysisReport,
    ImageFindings,
    ImageQualityFlags,
    IssueType,
    ObjectType,
    Severity,
)
from app.pipeline.vlm_backend import analyze_image
from config.settings import settings

logger = logging.getLogger(__name__)


_PROMPTS: dict[str, str] = {
    "car": "Analyse this car image for damage. Return JSON with: detected_object, damage_present, damage_description, damage_region (bumper/door/hood/windshield/headlight/taillight/side_mirror/body/other), severity (LOW/MEDIUM/HIGH), quality_notes, manipulation_concerns.",
    "laptop": "Analyse this laptop image for damage. Return JSON with: detected_object, damage_present, damage_description, damage_region (screen/hinge/keyboard/trackpad/body/corner/port/other), severity (LOW/MEDIUM/HIGH), quality_notes, manipulation_concerns.",
    "package": "Analyse this package image for damage. Return JSON with: detected_object, damage_present, damage_description, damage_region (corner/side/seal/label/contents/other), severity (LOW/MEDIUM/HIGH), quality_notes, manipulation_concerns.",
    "unknown": "Analyse this image for physical damage. Return JSON with: detected_object, damage_present, damage_description, damage_region, severity (LOW/MEDIUM/HIGH), quality_notes, manipulation_concerns.",
}


def analyze_images(
    user_id: str,
    image_paths_str: str,
    expected_object_type: ObjectType,
    images_base_dir: str = ".",
) -> ImageAnalysisReport:
    """
    Analyse all submitted images for a claim.

    Args:
        user_id:              Claim user ID.
        image_paths_str:      Semicolon-separated relative image paths from CSV.
        expected_object_type: Object type from claim_object column.
        images_base_dir:      Base directory for resolving relative image paths.

    Returns:
        ImageAnalysisReport with per-image findings.
    """
    raw_paths = [p.strip() for p in image_paths_str.split(";") if p.strip()]
    seen_hashes: set[str] = set()
    findings: List[ImageFindings] = []

    for raw_path in raw_paths:
        # Extract img_N identifier from path
        image_id = _extract_image_id(raw_path)

        # Resolve full path
        full_path = str(Path(images_base_dir) / raw_path)
        file_exists = Path(full_path).exists()

        if not file_exists:
            # Images not provided in dataset — generate a synthetic finding
            finding = _synthetic_finding(image_id, expected_object_type, raw_path)
            findings.append(finding)
            logger.debug("Image not found at %s — using synthetic finding.", full_path)
            continue

        # Quality checks
        quality = _check_image_quality(full_path)

        # Duplicate detection
        phash = _perceptual_hash(full_path)
        if phash in seen_hashes:
            quality = quality.model_copy(update={"quality_notes": (quality.quality_notes or "") + " [duplicate]"})
        seen_hashes.add(phash)

        # VLM / heuristic analysis
        prompt = _PROMPTS.get(expected_object_type.value, _PROMPTS["unknown"])
        try:
            raw_response = analyze_image(full_path, prompt)
            data = _parse_vlm_response(raw_response)
        except Exception as e:
            logger.error("Image %s VLM error: %s", image_id, e)
            data = {}

        if data.get("manipulation_concerns"):
            quality = quality.model_copy(update={"possible_manipulation": True})
        if data.get("quality_notes") and "text_instruction" in str(data.get("quality_notes", "")):
            quality = quality.model_copy(update={"text_instruction_detected": True})

        finding = ImageFindings(
            image_id=image_id,
            detected_object_type=_parse_object_type(data.get("detected_object", "unknown")),
            detected_damage_present=bool(data.get("damage_present", False)),
            detected_damage_description=data.get("damage_description", "No description available"),
            detected_damage_region=data.get("damage_region", "unspecified"),
            severity_estimate=_parse_severity(data.get("severity", "LOW")),
            is_valid_image=bool(data.get("is_valid_image", True)),
            quality=quality,
            raw_vlm_response=raw_response,
        )
        findings.append(finding)

    return ImageAnalysisReport(user_id=user_id, findings=findings)


def _synthetic_finding(image_id: str, obj_type: ObjectType, raw_path: str) -> ImageFindings:
    """
    When images are not physically present (HackerRank dataset has no images),
    produce a neutral finding that lets downstream logic decide on text evidence.
    """
    return ImageFindings(
        image_id=image_id,
        detected_object_type=obj_type,
        detected_damage_present=True,  # assume damage present — let matcher decide
        detected_damage_description="Image not available for analysis; decision based on claim text",
        detected_damage_region="unspecified",
        severity_estimate=Severity.MEDIUM,
        is_valid_image=True,
        quality=ImageQualityFlags(quality_notes="Image file not present in environment"),
    )


def _extract_image_id(path: str) -> str:
    """Extract img_N from paths like images/test/case_001/img_1.jpg → img_1"""
    stem = Path(path).stem  # e.g. "img_1"
    if re.match(r"img_\d+", stem):
        return stem
    return stem


def _check_image_quality(image_path: str) -> ImageQualityFlags:
    img = cv2.imread(image_path)
    if img is None:
        return ImageQualityFlags(quality_notes="Could not decode image")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness = float(hsv[:, :, 2].mean())
    is_blurry = blur_score < settings.BLUR_THRESHOLD
    is_dark = brightness < settings.DARK_THRESHOLD
    is_low_res = (w * h) < settings.MIN_PIXEL_COUNT
    edge_strip = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    is_partial = float(np.count_nonzero(edge_strip)) / len(edge_strip) > 0.95

    return ImageQualityFlags(
        is_blurry=is_blurry,
        is_dark=is_dark,
        is_low_resolution=is_low_res,
        is_partial=is_partial,
        quality_notes=f"blur={blur_score:.1f} brightness={brightness:.1f} res={w}x{h}",
    )


def _perceptual_hash(image_path: str) -> str:
    try:
        img = Image.open(image_path).convert("L").resize((8, 8), Image.LANCZOS)
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        bits = "".join("1" if p >= avg else "0" for p in pixels)
        return hashlib.md5(bits.encode()).hexdigest()
    except Exception:
        return hashlib.md5(open(image_path, "rb").read()).hexdigest()


def _parse_vlm_response(text: str) -> dict:
    clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("```").strip()
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {}


def _parse_object_type(value: str) -> ObjectType:
    val = (value or "").lower().strip()
    return {"car": ObjectType.CAR, "laptop": ObjectType.LAPTOP,
            "package": ObjectType.PACKAGE}.get(val, ObjectType.UNKNOWN)


def _parse_severity(value: str) -> Severity:
    val = (value or "").upper().strip()
    mapping = {"LOW": Severity.LOW, "MEDIUM": Severity.MEDIUM, "HIGH": Severity.HIGH}
    return mapping.get(val, Severity.MEDIUM)
