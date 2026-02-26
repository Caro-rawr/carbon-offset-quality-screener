"""
test_red_flags.py — tests for RedFlagDetector
These tests overlap with test_scorer.py's flag tests but focus exclusively on red_flags.py
"""
import pytest
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scorer import CarbonQualityScorer
from src.red_flags import RedFlagDetector, FLAG_CATALOGUE
from datetime import datetime


@pytest.fixture
def scored_sample():
    scorer = CarbonQualityScorer()
    df = pd.DataFrame({
        "project_id": ["VCS001", "VCS002"],
        "name": ["Forest A", "Solar B"],
        "country": ["Cambodia", "United States"],
        "project_type": ["REDD+", "Renewable Energy"],
        "status": ["Registered", "Registered"],
        "registration_date": pd.to_datetime(["2008-01-01", "2022-01-01"]),
        "crediting_period_start": pd.to_datetime(["2006-01-01", "2021-01-01"]),
        "crediting_period_end": pd.to_datetime(["2023-06-01", "2041-01-01"]),
        "total_issued": [100_000_000, 2_000_000],
        "total_retired": [0, 1_900_000],
        "total_buffer_pool": [10_000_000, 200_000],
        "total_cancelled": [0, 0],
        "estimated_annual_reductions": [10_000_000, 200_000],
        "proponent": ["Org A", "Org B"],
        "region": ["Asia Pacific", "North America"],
    })
    return scorer.score_all(df, reference_date=datetime(2024, 1, 1))


class TestRedFlagDetector:

    def test_catalogue_not_empty(self):
        assert len(FLAG_CATALOGUE) > 0

    def test_all_catalogue_flags_have_severity(self):
        for code, flag in FLAG_CATALOGUE.items():
            assert flag.severity in ("high", "medium", "low"), f"{code} has invalid severity"

    def test_detect_returns_flags_column(self, scored_sample):
        detector = RedFlagDetector()
        result = detector.detect(scored_sample)
        assert "flags" in result.columns
        assert "flag_count" in result.columns
        assert "max_severity" in result.columns

    def test_high_risk_project_has_multiple_flags(self, scored_sample):
        detector = RedFlagDetector()
        result = detector.detect(scored_sample)
        redd = result[result["project_id"] == "VCS001"]["flag_count"].iloc[0]
        assert redd >= 3  # REDD_CONTROVERSY + MASSIVE_ISSUANCE + ZERO_RETIREMENTS + HIGH_VINTAGE + WEAK_GOVERNANCE

    def test_clean_project_has_fewer_flags(self, scored_sample):
        detector = RedFlagDetector()
        result = detector.detect(scored_sample)
        redd_flags = result[result["project_id"] == "VCS001"]["flag_count"].iloc[0]
        solar_flags = result[result["project_id"] == "VCS002"]["flag_count"].iloc[0]
        assert solar_flags < redd_flags

    def test_max_severity_high_for_risky_project(self, scored_sample):
        detector = RedFlagDetector()
        result = detector.detect(scored_sample)
        sev = result[result["project_id"] == "VCS001"]["max_severity"].iloc[0]
        assert sev == "high"

    def test_get_flag_summary_structure(self, scored_sample):
        detector = RedFlagDetector()
        flagged = detector.detect(scored_sample)
        summary = detector.get_flag_summary(flagged)
        assert isinstance(summary, pd.DataFrame)
        assert "flag_code" in summary.columns
        assert "severity" in summary.columns
        assert "pct_of_portfolio" in summary.columns
