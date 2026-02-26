"""
tests/test_scorer.py
Unit tests for CarbonQualityScorer and RedFlagDetector.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scorer import CarbonQualityScorer
from src.red_flags import RedFlagDetector


@pytest.fixture
def sample_df():
    """Minimal synthetic DataFrame for testing scorer logic."""
    return pd.DataFrame({
        "project_id": ["VCS001", "VCS002", "VCS003", "VCS004"],
        "name": ["Forest A", "Solar B", "REDD C", "Methane D"],
        "country": ["Brazil", "India", "Indonesia", "United States"],
        "project_type": [
            "Improved Forest Management",
            "Renewable Energy",
            "REDD+",
            "Methane Capture - Livestock"
        ],
        "status": ["Registered"] * 4,
        "registration_date": pd.to_datetime([
            "2015-01-01", "2022-06-01", "2010-03-15", "2021-09-01"
        ]),
        "crediting_period_start": pd.to_datetime([
            "2014-01-01", "2021-01-01", "2008-01-01", "2020-01-01"
        ]),
        "crediting_period_end": pd.to_datetime([
            "2029-01-01", "2041-01-01", "2023-01-01", "2040-01-01"
        ]),
        "total_issued": [5_000_000, 2_000_000, 80_000_000, 1_500_000],
        "total_retired": [3_000_000, 1_800_000, 2_000_000, 0],
        "total_buffer_pool": [500_000, 200_000, 8_000_000, 150_000],
        "total_cancelled": [0, 0, 0, 0],
        "estimated_annual_reductions": [500_000, 200_000, 8_000_000, 150_000],
        "proponent": ["Org A", "Org B", "Org C", "Org D"],
        "region": ["Latin America", "Asia Pacific", "Asia Pacific", "North America"],
    })


class TestCarbonQualityScorer:

    def setup_method(self):
        self.scorer = CarbonQualityScorer()
        self.ref_date = datetime(2024, 1, 1)

    def test_score_all_returns_cqi_column(self, sample_df):
        result = self.scorer.score_all(sample_df, reference_date=self.ref_date)
        assert "cqi" in result.columns

    def test_cqi_range(self, sample_df):
        result = self.scorer.score_all(sample_df, reference_date=self.ref_date)
        assert (result["cqi"] >= 0).all()
        assert (result["cqi"] <= 100).all()

    def test_quality_tier_assigned(self, sample_df):
        result = self.scorer.score_all(sample_df, reference_date=self.ref_date)
        assert "quality_tier" in result.columns
        assert result["quality_tier"].notna().all()

    def test_redd_type_lower_score(self, sample_df):
        result = self.scorer.score_all(sample_df, reference_date=self.ref_date)
        redd_score = result.loc[result["project_type"] == "REDD+", "project_type_score"].iloc[0]
        renewable_score = result.loc[result["project_type"] == "Renewable Energy", "project_type_score"].iloc[0]
        assert redd_score < renewable_score

    def test_high_retirement_ratio_scores_well(self, sample_df):
        # VCS002 Solar B has 90% retirement rate
        result = self.scorer.score_all(sample_df, reference_date=self.ref_date)
        solar = result.loc[result["project_id"] == "VCS002", "retirement_ratio_score"].iloc[0]
        methane = result.loc[result["project_id"] == "VCS004", "retirement_ratio_score"].iloc[0]
        assert solar > methane

    def test_older_project_lower_vintage_score(self, sample_df):
        result = self.scorer.score_all(sample_df, reference_date=self.ref_date)
        old_score = result.loc[result["project_id"] == "VCS003", "vintage_score"].iloc[0]
        new_score = result.loc[result["project_id"] == "VCS002", "vintage_score"].iloc[0]
        assert new_score > old_score

    def test_weights_sum_to_one(self):
        total = sum(self.scorer.WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9


class TestRedFlagDetector:

    def setup_method(self):
        self.scorer = CarbonQualityScorer()
        self.detector = RedFlagDetector()
        self.ref_date = datetime(2024, 1, 1)

    def test_flags_column_created(self, sample_df):
        scored = self.scorer.score_all(sample_df, reference_date=self.ref_date)
        flagged = self.detector.detect(scored)
        assert "flags" in flagged.columns
        assert "flag_count" in flagged.columns

    def test_zero_retirements_flagged(self, sample_df):
        scored = self.scorer.score_all(sample_df, reference_date=self.ref_date)
        flagged = self.detector.detect(scored)
        methane_flags = flagged.loc[flagged["project_id"] == "VCS004", "flags"].iloc[0]
        assert "ZERO_RETIREMENTS" in methane_flags

    def test_redd_controversy_flagged(self, sample_df):
        scored = self.scorer.score_all(sample_df, reference_date=self.ref_date)
        flagged = self.detector.detect(scored)
        redd_flags = flagged.loc[flagged["project_id"] == "VCS003", "flags"].iloc[0]
        assert "REDD_CONTROVERSY" in redd_flags

    def test_massive_issuance_flagged(self, sample_df):
        scored = self.scorer.score_all(sample_df, reference_date=self.ref_date)
        flagged = self.detector.detect(scored)
        redd_flags = flagged.loc[flagged["project_id"] == "VCS003", "flags"].iloc[0]
        assert "MASSIVE_ISSUANCE" in redd_flags

    def test_flag_summary_returns_dataframe(self, sample_df):
        scored = self.scorer.score_all(sample_df, reference_date=self.ref_date)
        flagged = self.detector.detect(scored)
        summary = self.detector.get_flag_summary(flagged)
        assert isinstance(summary, pd.DataFrame)
        assert "flag_code" in summary.columns

    def test_clean_project_has_fewer_flags(self, sample_df):
        scored = self.scorer.score_all(sample_df, reference_date=self.ref_date)
        flagged = self.detector.detect(scored)
        solar_flags = flagged.loc[flagged["project_id"] == "VCS002", "flag_count"].iloc[0]
        redd_flags = flagged.loc[flagged["project_id"] == "VCS003", "flag_count"].iloc[0]
        assert solar_flags < redd_flags
