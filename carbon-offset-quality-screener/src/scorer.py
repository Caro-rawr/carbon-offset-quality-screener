"""
scorer.py
Composite Quality Index (CQI) engine for voluntary carbon offset projects.

Each scoring dimension is calculated independently and combined into a
0–100 weighted composite score. Higher = better quality / lower risk.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

# High-risk project types based on literature and market signals (Berkeley Carbon Trading Project,
# CarbonPlan, ICVCM review processes)
HIGH_RISK_TYPES = {
    "REDD+",
    "Avoided Deforestation",
    "Avoided Unplanned Deforestation and Degradation",
}

# Medium-risk project types
MEDIUM_RISK_TYPES = {
    "Improved Forest Management",
    "Agriculture Forestry and Other Land Use",
    "Afforestation/Reforestation",
}

# Country-level governance quality proxy (0=poor, 1=high)
# Based on World Governance Indicators tiers (simplified for scoring)
COUNTRY_GOVERNANCE_SCORE = {
    "Brazil": 0.55, "Indonesia": 0.50, "Peru": 0.60, "Colombia": 0.55,
    "Mexico": 0.60, "Kenya": 0.50, "Tanzania": 0.45, "Cambodia": 0.40,
    "India": 0.55, "China": 0.50, "Vietnam": 0.45, "Madagascar": 0.35,
    "Democratic Republic of the Congo": 0.25, "Uganda": 0.40,
    "Chile": 0.75, "Costa Rica": 0.75, "Uruguay": 0.75,
    "Ghana": 0.55, "Senegal": 0.55, "Rwanda": 0.60,
    "United States": 0.85, "Canada": 0.85, "Australia": 0.85,
    "Germany": 0.90, "Sweden": 0.90,
}
DEFAULT_GOVERNANCE_SCORE = 0.50  # for countries not in the map


class CarbonQualityScorer:
    """
    Scores each project on 6 dimensions and computes a weighted CQI.
    
    All sub-scores are normalized to [0, 100] before weighting.
    
    Weights:
        - vintage_score:         0.20
        - retirement_ratio_score: 0.20
        - project_type_score:    0.20
        - transparency_score:    0.15
        - additionality_score:   0.15
        - governance_score:      0.10
    """

    WEIGHTS = {
        "vintage_score": 0.20,
        "retirement_ratio_score": 0.20,
        "project_type_score": 0.20,
        "transparency_score": 0.15,
        "additionality_score": 0.15,
        "governance_score": 0.10,
    }

    def score_all(self, df: pd.DataFrame, reference_date: datetime = None) -> pd.DataFrame:
        """
        Applies all scoring dimensions to a dataframe of projects.
        Returns the original dataframe with score columns appended.
        
        Args:
            df: Cleaned project dataframe from VerraDataLoader
            reference_date: Scoring reference date (defaults to today)
        
        Returns:
            DataFrame with CQI and dimension scores added
        """
        if reference_date is None:
            reference_date = datetime.today()

        df = df.copy()

        df["vintage_score"] = df.apply(
            lambda r: self._vintage_score(r, reference_date), axis=1
        )
        df["retirement_ratio_score"] = df.apply(
            self._retirement_ratio_score, axis=1
        )
        df["project_type_score"] = df["project_type"].apply(
            self._project_type_score
        )
        df["transparency_score"] = df.apply(
            self._transparency_score, axis=1
        )
        df["additionality_score"] = df.apply(
            lambda r: self._additionality_score(r, reference_date), axis=1
        )
        df["governance_score"] = df["country"].apply(
            self._governance_score
        )

        # Weighted composite
        df["cqi"] = sum(
            df[dim] * weight
            for dim, weight in self.WEIGHTS.items()
        ).round(2)

        # Quality tier classification
        df["quality_tier"] = pd.cut(
            df["cqi"],
            bins=[0, 40, 55, 70, 85, 100],
            labels=["Very Low", "Low", "Medium", "High", "Very High"],
            include_lowest=True
        )

        logger.info(f"Scored {len(df)} projects. CQI range: {df['cqi'].min():.1f} – {df['cqi'].max():.1f}")
        return df

    # ------------------------------------------------------------------ #
    # Dimension scoring methods                                            #
    # ------------------------------------------------------------------ #

    def _vintage_score(self, row: pd.Series, reference_date: datetime) -> float:
        """
        Penalizes projects with older registration dates.
        
        Logic: Credits from projects registered >8 years ago face market
        discount. Score decays linearly from 100 (recent) to 20 (>12 years).
        """
        reg_date = row.get("registration_date")
        if pd.isna(reg_date):
            return 50.0  # neutral when unknown

        age_years = (reference_date - pd.Timestamp(reg_date)).days / 365.25

        if age_years <= 3:
            return 100.0
        elif age_years <= 8:
            # Linear decay: 100 → 70
            return 100.0 - (age_years - 3) * 6.0
        elif age_years <= 12:
            # Steeper decay: 70 → 30
            return 70.0 - (age_years - 8) * 10.0
        else:
            return max(10.0, 30.0 - (age_years - 12) * 5.0)

    def _retirement_ratio_score(self, row: pd.Series) -> float:
        """
        Rewards projects with high credit retirement rates.
        
        Logic: High retirement-to-issuance ratio signals actual demand and use.
        Ratio >= 80% earns 100; ratio < 5% earns near 0.
        """
        issued = row.get("total_issued", 0)
        retired = row.get("total_retired", 0)

        if issued <= 0:
            return 50.0  # neutral when no issuance data

        ratio = retired / issued

        if ratio >= 0.80:
            return 100.0
        elif ratio >= 0.50:
            return 60.0 + (ratio - 0.50) * 200.0
        elif ratio >= 0.20:
            return 30.0 + (ratio - 0.20) * 100.0
        elif ratio >= 0.05:
            return 10.0 + (ratio - 0.05) * 133.0
        else:
            return max(0.0, ratio * 200.0)

    def _project_type_score(self, project_type: str) -> float:
        """
        Applies a risk discount based on project type.
        
        Logic: REDD+ and avoided deforestation have well-documented permanence
        and additionality controversies (CarbonPlan 2023, Berkeley Carbon Trading
        Project). Lower score = higher controversy risk.
        """
        if pd.isna(project_type):
            return 50.0

        pt = str(project_type).strip()

        if any(ht.lower() in pt.lower() for ht in HIGH_RISK_TYPES):
            return 30.0
        elif any(mt.lower() in pt.lower() for mt in MEDIUM_RISK_TYPES):
            return 60.0
        else:
            # Renewable energy, methane capture, etc. — lower permanence risk
            return 85.0

    def _transparency_score(self, row: pd.Series) -> float:
        """
        Rewards data completeness as a proxy for documentation quality.
        
        Logic: More complete public data = more auditable project. Each
        populated field contributes to the score.
        """
        scored_fields = [
            "proponent", "region", "crediting_period_start",
            "crediting_period_end", "estimated_annual_reductions",
            "total_buffer_pool"
        ]
        populated = sum(
            1 for f in scored_fields
            if f in row and not pd.isna(row.get(f)) and row.get(f) != 0
        )
        return round((populated / len(scored_fields)) * 100, 1)

    def _additionality_score(self, row: pd.Series, reference_date: datetime) -> float:
        """
        Proxy for additionality based on project registration timing.
        
        Logic: Projects registered quickly after crediting period start are
        more likely to be genuinely additional (activity wasn't happening already).
        Large gaps between activity start and registration are a weak additionality signal.
        """
        reg_date = row.get("registration_date")
        cp_start = row.get("crediting_period_start")

        if pd.isna(reg_date) or pd.isna(cp_start):
            return 50.0

        lag_years = (pd.Timestamp(reg_date) - pd.Timestamp(cp_start)).days / 365.25

        if lag_years <= 1:
            return 90.0
        elif lag_years <= 3:
            return 75.0
        elif lag_years <= 6:
            return 55.0
        elif lag_years <= 10:
            return 35.0
        else:
            return 15.0

    def _governance_score(self, country: str) -> float:
        """
        Applies a governance quality proxy from the country-level lookup table.
        
        Returns a 0–100 score based on World Governance Indicators approximation.
        """
        if pd.isna(country):
            return DEFAULT_GOVERNANCE_SCORE * 100

        score = COUNTRY_GOVERNANCE_SCORE.get(str(country).strip(), DEFAULT_GOVERNANCE_SCORE)
        return round(score * 100, 1)
