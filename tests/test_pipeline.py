"""
tests/test_pipeline.py

Unit tests for each pipeline stage using mocked VLM responses.
Tests cover happy paths, edge cases, and the critical rule that
history cannot override visual evidence.

Run: pytest tests/ -v
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from app.models.schemas import (
    ClaimExtraction, ClaimStatus, ImageAnalysisReport, ImageFindings,
    ImageQualityFlags, MatchResult, ObjectType, PerImageVerdict,
    RiskFlag, RiskReport, Severity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def car_claim():
    return ClaimExtraction(
        claim_id="TEST001",
        user_id="USER001",
        object_type=ObjectType.CAR,
        claimed_damage="Large dent on the front bumper",
        claimed_damage_type="dent",
        claimed_object_part="bumper",
        raw_conversation="I have a dent on my front bumper.",
    )


@pytest.fixture
def good_image_finding():
    return ImageFindings(
        image_id="bumper_photo.jpg",
        detected_object_type=ObjectType.CAR,
        detected_damage_present=True,
        detected_damage_description="A deep dent approximately 15cm wide on the front bumper",
        detected_damage_region="bumper",
        severity_estimate=Severity.MEDIUM,
        quality=ImageQualityFlags(is_blurry=False, is_dark=False),
    )


@pytest.fixture
def blurry_image_finding():
    return ImageFindings(
        image_id="blurry.jpg",
        detected_object_type=ObjectType.CAR,
        detected_damage_present=True,
        detected_damage_description="Possible damage, unclear due to blur",
        detected_damage_region="bumper",
        severity_estimate=Severity.LOW,
        quality=ImageQualityFlags(is_blurry=True, quality_notes="blur=12.4 brightness=130"),
    )


@pytest.fixture
def good_report(car_claim, good_image_finding):
    return ImageAnalysisReport(claim_id=car_claim.claim_id, findings=[good_image_finding])


@pytest.fixture
def blurry_report(car_claim, blurry_image_finding):
    return ImageAnalysisReport(claim_id=car_claim.claim_id, findings=[blurry_image_finding])


# ---------------------------------------------------------------------------
# Claim Parser tests
# ---------------------------------------------------------------------------

class TestClaimParser:
    @patch("app.pipeline.claim_parser._call_text_model")
    def test_extract_car_claim(self, mock_llm):
        mock_llm.return_value = '''{
            "object_type": "car",
            "claimed_damage": "dent on bumper",
            "claimed_damage_type": "dent",
            "claimed_object_part": "front bumper",
            "additional_details": null
        }'''
        from app.pipeline.claim_parser import parse_claim
        result = parse_claim("CLM001", "USR001", "My front bumper has a dent.")
        assert result.object_type == ObjectType.CAR
        assert result.claimed_damage_type == "dent"
        assert result.claimed_object_part == "front bumper"

    @patch("app.pipeline.claim_parser._call_text_model")
    def test_fallback_on_bad_json(self, mock_llm):
        mock_llm.return_value = "Sorry, I cannot help with that."
        from app.pipeline.claim_parser import parse_claim
        # Should not raise — fallback keyword extraction kicks in
        result = parse_claim("CLM002", "USR001", "My laptop screen is cracked.")
        assert result.object_type == ObjectType.LAPTOP

    @patch("app.pipeline.claim_parser._call_text_model")
    def test_unknown_object_type(self, mock_llm):
        mock_llm.return_value = '{"object_type": "unknown", "claimed_damage": "broken", ' \
                                 '"claimed_damage_type": "other", "claimed_object_part": "unknown"}'
        from app.pipeline.claim_parser import parse_claim
        result = parse_claim("CLM003", "USR001", "Something is broken.")
        assert result.object_type == ObjectType.UNKNOWN


# ---------------------------------------------------------------------------
# Evidence Matcher tests — critical rule checks
# ---------------------------------------------------------------------------

class TestEvidenceMatcher:
    @patch("app.pipeline.evidence_matcher._call_match_vlm")
    def test_supported_when_image_matches(self, mock_vlm, car_claim, good_report):
        mock_vlm.return_value = '{"verdict": "SUPPORTED", "confidence": 0.9, ' \
                                 '"notes": "Dent visible on bumper."}'
        from app.pipeline.evidence_matcher import match_evidence
        result = match_evidence(car_claim, good_report)
        assert result.overall_verdict == ClaimStatus.supported
        assert "bumper_photo.jpg" in result.supporting_image_ids

    def test_insufficient_when_image_blurry(self, car_claim, blurry_report):
        from app.pipeline.evidence_matcher import match_evidence
        result = match_evidence(car_claim, blurry_report)
        # Blurry image → insufficient, no VLM call needed
        assert result.overall_verdict in (ClaimStatus.not_enough_information, ClaimStatus.contradicted)

    def test_contradicted_wrong_object(self, car_claim):
        wrong_finding = ImageFindings(
            image_id="laptop.jpg",
            detected_object_type=ObjectType.LAPTOP,   # wrong!
            detected_damage_present=False,
            detected_damage_description="This is a laptop, not a car",
            detected_damage_region="screen",
            severity_estimate=Severity.LOW,
            quality=ImageQualityFlags(),
        )
        report = ImageAnalysisReport(claim_id="TEST001", findings=[wrong_finding])
        from app.pipeline.evidence_matcher import match_evidence
        result = match_evidence(car_claim, report)
        assert result.overall_verdict == ClaimStatus.contradicted

    def test_empty_images_gives_insufficient(self, car_claim):
        empty_report = ImageAnalysisReport(claim_id="TEST001", findings=[])
        from app.pipeline.evidence_matcher import match_evidence
        result = match_evidence(car_claim, empty_report)
        assert result.overall_verdict == ClaimStatus.not_enough_information


# ---------------------------------------------------------------------------
# Risk Detector tests — history cannot override visual evidence
# ---------------------------------------------------------------------------

class TestRiskDetector:
    def test_suspicious_history_adds_flag(self, car_claim, good_report):
        import pandas as pd
        from app.pipeline.risk_detector import detect_risks

        # User with 6 claims in 90 days (above threshold of 5)
        history = pd.DataFrame([
            {"user_id": "USER001", "claim_date": pd.Timestamp.now() - pd.Timedelta(days=i*10),
             "decision": "SUPPORTED", "object_type": "car"}
            for i in range(6)
        ])

        match = MatchResult(
            claim_id="TEST001",
            overall_verdict=ClaimStatus.supported,
            per_image_verdicts=[
                PerImageVerdict(image_id="bumper_photo.jpg", verdict=ClaimStatus.supported,
                                confidence=0.9, match_notes="Match")
            ],
            supporting_image_ids=["bumper_photo.jpg"],
            match_summary="Supported",
        )
        risk = detect_risks("TEST001", "USER001", good_report, match, history)
        assert RiskFlag.SUSPICIOUS_USER_HISTORY in risk.risk_flags
        # But this ALONE shouldn't flip the decision (tested in Decision Engine)

    def test_clean_history_no_flag(self, car_claim, good_report):
        import pandas as pd
        from app.pipeline.risk_detector import detect_risks

        history = pd.DataFrame([
            {"user_id": "USER001", "claim_date": pd.Timestamp("2023-01-01"),
             "decision": "SUPPORTED", "object_type": "car"}
        ])
        match = MatchResult(
            claim_id="TEST001", overall_verdict=ClaimStatus.supported,
            per_image_verdicts=[], supporting_image_ids=[], match_summary=""
        )
        risk = detect_risks("TEST001", "USER001", good_report, match, history)
        assert RiskFlag.SUSPICIOUS_USER_HISTORY not in risk.risk_flags


# ---------------------------------------------------------------------------
# Decision Engine tests — key invariant checks
# ---------------------------------------------------------------------------

class TestDecisionEngine:
    def test_supported_decision_not_overridden_by_history_alone(
        self, car_claim, good_report, good_image_finding
    ):
        from app.pipeline.decision_engine import make_decision

        match = MatchResult(
            claim_id="TEST001",
            overall_verdict=ClaimStatus.supported,
            per_image_verdicts=[
                PerImageVerdict(image_id="bumper_photo.jpg", verdict=ClaimStatus.supported,
                                confidence=0.95, match_notes="Clear dent visible.")
            ],
            supporting_image_ids=["bumper_photo.jpg"],
            match_summary="Supported",
        )
        # History risk alone should NOT change SUPPORTED to CONTRADICTED
        risk = RiskReport(
            claim_id="TEST001",
            risk_flags=[RiskFlag.SUSPICIOUS_USER_HISTORY],
            risk_score=0.3,
            suspicious_user_history=True,
        )
        result = make_decision(car_claim, good_report, match, risk)
        # Must still be SUPPORTED (history doesn't override visual evidence)
        assert result.decision == ClaimStatus.supported

    def test_manipulation_flag_downgrades_supported(self, car_claim, good_report):
        from app.pipeline.decision_engine import make_decision

        match = MatchResult(
            claim_id="TEST001",
            overall_verdict=ClaimStatus.supported,
            per_image_verdicts=[
                PerImageVerdict(image_id="bumper_photo.jpg", verdict=ClaimStatus.supported,
                                confidence=0.9, match_notes="Dent visible.")
            ],
            supporting_image_ids=["bumper_photo.jpg"],
            match_summary="Supported",
        )
        risk = RiskReport(
            claim_id="TEST001",
            risk_flags=[RiskFlag.POTENTIAL_MANIPULATION],
            risk_score=0.4,
            suspicious_user_history=False,
        )
        result = make_decision(car_claim, good_report, match, risk)
        # Manipulation concern should downgrade to INSUFFICIENT_EVIDENCE
        assert result.decision == ClaimStatus.not_enough_information

    def test_severity_taken_from_supporting_images(self, car_claim, good_report):
        from app.pipeline.decision_engine import make_decision

        match = MatchResult(
            claim_id="TEST001",
            overall_verdict=ClaimStatus.supported,
            per_image_verdicts=[],
            supporting_image_ids=["bumper_photo.jpg"],
            match_summary="Supported",
        )
        risk = RiskReport(claim_id="TEST001", risk_flags=[], risk_score=0.0,
                          suspicious_user_history=False)
        result = make_decision(car_claim, good_report, match, risk)
        assert result.severity == Severity.MEDIUM  # matches good_image_finding
