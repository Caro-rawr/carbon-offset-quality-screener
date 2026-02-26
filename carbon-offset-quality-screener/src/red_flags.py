"""
red_flags.py
Detects specific integrity risk signals in carbon offset projects.
Each flag is independent; a project can have multiple flags.
"""

import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class RedFlag:
    code: str
    label: str
    description: str
    severity: str  # "high", "medium", "low"


# Catalogue of defined flags
FLAG_CATALOGUE = {
    "HIGH_VINTAGE": RedFlag(
        code="HIGH_VINTAGE",
        label="High Vintage Age",
        description="Project registered more than 10 years ago. Credits may face market discount.",
        severity="medium"
    ),
    "ZERO_RETIREMENTS": RedFlag(
        code="ZERO_RETIREMENTS",
        label="No Retirements Recorded",
        description="Project has issued credits but shows zero retirements — potential demand signal.",
        severity="high"
    ),
    "LOW_RETIREMENT_RATIO": RedFlag(
        code="LOW_RETIREMENT_RATIO",
        label="Low Retirement Rate (<10%)",
        description="Less than 10% of issued credits have been retired.",
        severity="medium"
    ),
    "REDD_CONTROVERSY": RedFlag(
        code="REDD_CONTROVERSY",
        label="REDD+ Controversy Risk",
        description="REDD+ projects face scrutiny over permanence and additionality (CarbonPlan, Berkeley CPT).",
        severity="high"
    ),
    "MASSIVE_ISSUANCE": RedFlag(
        code="MASSIVE_ISSUANCE",
        label="Unusually High Issuance Volume",
        description="Total issuance exceeds 50M tCO2e — may indicate inflated baselines.",
        severity="medium"
    ),
    "REGISTRATION_LAG": RedFlag(
        code="REGISTRATION_LAG",
        label="Long Registration Lag (>5 years)",
        description="Significant delay between crediting period start and registration date.",
        severity="medium"
    ),
    "WEAK_GOVERNANCE": RedFlag(
        code="WEAK_GOVERNANCE",
        label="Weak Host Country Governance",
        description="Project located in a jurisdiction with low governance quality score (<0.45).",
        severity="medium"
    ),
    "EXPIRED_CREDITING": RedFlag(
        code="EXPIRED_CREDITING",
        label="Expired or Expiring Crediting Period",
        description="Crediting period has ended or ends within 12 months.",
        severity="low"
    ),
    "INCOMPLETE_DATA": RedFlag(
        code="INCOMPLETE_DATA",
        label="Incomplete Public Documentation",
        description="Transparency score below 40 — key project fields missing from registry.",
        severity="low"
    ),
}


class RedFlagDetector:
    """
    Applies all flag detection rules to a scored projects dataframe.
    
    Adds two columns:
        - 'flags': list of RedFlag codes triggered for each project
        - 'flag_count': total number of flags
        - 'max_severity': worst severity level among triggered flags
    """

    GOVERNANCE_THRESHOLD = 45.0  # governance_score below this triggers flag

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs all flag checks on scored dataframe.
        
        Requires columns produced by CarbonQualityScorer (scorer.py).
        """
        df = df.copy()
        df["flags"] = [[] for _ in range(len(df))]

        df = self._flag_high_vintage(df)
        df = self._flag_zero_retirements(df)
        df = self._flag_low_retirement_ratio(df)
        df = self._flag_redd_controversy(df)
        df = self._flag_massive_issuance(df)
        df = self._flag_registration_lag(df)
        df = self._flag_weak_governance(df)
        df = self._flag_expired_crediting(df)
        df = self._flag_incomplete_data(df)

        df["flag_count"] = df["flags"].apply(len)
        df["max_severity"] = df["flags"].apply(self._max_severity)

        logger.info(
            f"Flag detection complete. "
            f"Projects with ≥1 flag: {(df['flag_count'] > 0).sum()} / {len(df)}"
        )
        return df

    def get_flag_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns a summary table of flag frequency across the dataset."""
        from collections import Counter
        all_flags = [flag for flags in df["flags"] for flag in flags]
        counts = Counter(all_flags)
        rows = []
        for code, count in counts.most_common():
            cat = FLAG_CATALOGUE.get(code)
            rows.append({
                "flag_code": code,
                "label": cat.label if cat else code,
                "severity": cat.severity if cat else "unknown",
                "project_count": count,
                "pct_of_portfolio": round(count / len(df) * 100, 1)
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Individual flag rules                                                #
    # ------------------------------------------------------------------ #

    def _add_flag(self, df: pd.DataFrame, mask: pd.Series, code: str) -> pd.DataFrame:
        df.loc[mask, "flags"] = df.loc[mask, "flags"].apply(
            lambda x: x + [code] if code not in x else x
        )
        return df

    def _flag_high_vintage(self, df: pd.DataFrame) -> pd.DataFrame:
        if "vintage_score" in df.columns:
            mask = df["vintage_score"] < 30
            return self._add_flag(df, mask, "HIGH_VINTAGE")
        return df

    def _flag_zero_retirements(self, df: pd.DataFrame) -> pd.DataFrame:
        if "total_retired" in df.columns and "total_issued" in df.columns:
            mask = (df["total_issued"] > 0) & (df["total_retired"] == 0)
            return self._add_flag(df, mask, "ZERO_RETIREMENTS")
        return df

    def _flag_low_retirement_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        if "total_retired" in df.columns and "total_issued" in df.columns:
            ratio = df["total_retired"] / df["total_issued"].replace(0, np.nan)
            mask = (ratio < 0.10) & (df["total_issued"] > 0) & (df["total_retired"] > 0)
            return self._add_flag(df, mask, "LOW_RETIREMENT_RATIO")
        return df

    def _flag_redd_controversy(self, df: pd.DataFrame) -> pd.DataFrame:
        if "project_type" in df.columns:
            mask = df["project_type"].str.contains("REDD", case=False, na=False)
            return self._add_flag(df, mask, "REDD_CONTROVERSY")
        return df

    def _flag_massive_issuance(self, df: pd.DataFrame) -> pd.DataFrame:
        if "total_issued" in df.columns:
            mask = df["total_issued"] > 50_000_000
            return self._add_flag(df, mask, "MASSIVE_ISSUANCE")
        return df

    def _flag_registration_lag(self, df: pd.DataFrame) -> pd.DataFrame:
        if "additionality_score" in df.columns:
            mask = df["additionality_score"] < 40
            return self._add_flag(df, mask, "REGISTRATION_LAG")
        return df

    def _flag_weak_governance(self, df: pd.DataFrame) -> pd.DataFrame:
        if "governance_score" in df.columns:
            mask = df["governance_score"] < self.GOVERNANCE_THRESHOLD
            return self._add_flag(df, mask, "WEAK_GOVERNANCE")
        return df

    def _flag_expired_crediting(self, df: pd.DataFrame) -> pd.DataFrame:
        if "crediting_period_end" in df.columns:
            now = pd.Timestamp.today()
            threshold = now + pd.DateOffset(months=12)
            mask = df["crediting_period_end"].notna() & (df["crediting_period_end"] <= threshold)
            return self._add_flag(df, mask, "EXPIRED_CREDITING")
        return df

    def _flag_incomplete_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if "transparency_score" in df.columns:
            mask = df["transparency_score"] < 40
            return self._add_flag(df, mask, "INCOMPLETE_DATA")
        return df

    def _max_severity(self, flags: list) -> str:
        severity_order = {"high": 3, "medium": 2, "low": 1, "none": 0}
        if not flags:
            return "none"
        severities = [
            FLAG_CATALOGUE[f].severity for f in flags if f in FLAG_CATALOGUE
        ]
        if not severities:
            return "unknown"
        return max(severities, key=lambda s: severity_order.get(s, 0))
