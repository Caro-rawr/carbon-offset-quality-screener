"""
cleaner.py
----------
Standardizes, validates and enriches raw project data before scoring.
Handles missing values, type normalization and derived field calculation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict

logger = logging.getLogger(__name__)

# Project type normalization: maps API labels → canonical categories
TYPE_CANONICALIZATION = {
    "redd": "REDD+",
    "redd+": "REDD+",
    "improved forest management": "IFM",
    "afforestation": "ARR",
    "reforestation": "ARR",
    "revegetation": "ARR",
    "arr": "ARR",
    "agricultural land management": "ALM",
    "alm": "ALM",
    "wetland": "WRC",
    "wrc": "WRC",
    "renewable energy": "Renewable Energy",
    "wind": "Renewable Energy",
    "solar": "Renewable Energy",
    "hydro": "Renewable Energy",
    "methane": "Methane Capture",
    "landfill": "Methane Capture",
    "energy efficiency": "Energy Efficiency",
    "industrial": "Industrial",
    "ozone": "Industrial",
}


def clean_project_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline: impute, normalize, and derive fields.

    Parameters
    ----------
    df : pd.DataFrame
        Raw project data from fetcher.

    Returns
    -------
    pd.DataFrame
        Clean, enriched dataset ready for scoring.
    """
    df = df.copy()
    n_raw = len(df)

    # --- 1. Remove projects with no issuance data ---
    df = df[df.get("credits_issued_total", pd.Series(0)) > 0].copy()
    logger.info(f"Removed {n_raw - len(df)} projects with zero issuances.")

    # --- 2. Canonicalize project types ---
    if "project_type" in df.columns:
        df["project_type_raw"] = df["project_type"].copy()
        df["project_type"] = df["project_type"].apply(_canonicalize_type)

    # --- 3. Parse and validate dates ---
    if "registration_date" in df.columns:
        df["registration_date"] = pd.to_datetime(df["registration_date"], errors="coerce")

    # --- 4. Ensure numeric integrity ---
    numeric_cols = [
        "credits_issued_total",
        "credits_retired_total",
        "credits_in_buffer",
        "project_age_years",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    # Logical consistency: retired + buffer cannot exceed issued
    if all(c in df.columns for c in ["credits_retired_total", "credits_in_buffer", "credits_issued_total"]):
        total_accounted = df["credits_retired_total"] + df["credits_in_buffer"]
        overcounted = total_accounted > df["credits_issued_total"] * 1.01  # 1% tolerance
        if overcounted.any():
            logger.warning(
                f"{overcounted.sum()} projects have retired+buffer > issued. "
                "Capping retired at issued total."
            )
            df.loc[overcounted, "credits_retired_total"] = (
                df.loc[overcounted, "credits_issued_total"]
                - df.loc[overcounted, "credits_in_buffer"]
            ).clip(lower=0)

    # --- 5. Derive analytical fields ---
    df = _derive_fields(df)

    # --- 6. Impute missing project_age_years ---
    if "project_age_years" in df.columns:
        median_age = df["project_age_years"].median()
        missing_age = df["project_age_years"] == 0
        if missing_age.any():
            df.loc[missing_age, "project_age_years"] = median_age
            logger.info(f"Imputed project_age_years with median ({median_age:.1f} yr) for {missing_age.sum()} projects.")

    logger.info(f"Cleaned dataset: {len(df)} projects, {len(df.columns)} columns.")
    return df.reset_index(drop=True)


def _canonicalize_type(raw_type: str) -> str:
    """Map raw project type string to canonical category."""
    if pd.isna(raw_type):
        return "Unknown"
    lower = str(raw_type).lower().strip()
    for key, canonical in TYPE_CANONICALIZATION.items():
        if key in lower:
            return canonical
    return "Unknown"


def _derive_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Compute analytical derived fields used downstream in scoring."""

    # Retirement rate (0–1)
    if all(c in df.columns for c in ["credits_retired_total", "credits_issued_total"]):
        df["retirement_rate"] = np.where(
            df["credits_issued_total"] > 0,
            df["credits_retired_total"] / df["credits_issued_total"],
            0.0,
        ).clip(0, 1)

    # Buffer pool ratio (0–1)
    if all(c in df.columns for c in ["credits_in_buffer", "credits_issued_total"]):
        df["buffer_pool_ratio"] = np.where(
            df["credits_issued_total"] > 0,
            df["credits_in_buffer"] / df["credits_issued_total"],
            0.0,
        ).clip(0, 1)

    # Remaining credits (inventory)
    if all(c in df.columns for c in ["credits_issued_total", "credits_retired_total", "credits_in_buffer"]):
        df["credits_remaining"] = (
            df["credits_issued_total"]
            - df["credits_retired_total"]
            - df["credits_in_buffer"]
        ).clip(lower=0)

    # Size tier (for reporting)
    if "credits_issued_total" in df.columns:
        df["size_tier"] = pd.cut(
            df["credits_issued_total"],
            bins=[0, 100_000, 1_000_000, 10_000_000, float("inf")],
            labels=["Small (<100K)", "Medium (100K–1M)", "Large (1M–10M)", "XLarge (>10M)"],
        )

    return df


def compute_issuance_metrics(
    issuance_history: pd.DataFrame,
) -> Dict[str, float]:
    """
    From per-vintage issuance data, compute:
      - vintage_freshness_score: recency-weighted average vintage
      - issuance_cv: coefficient of variation in annual issuances
      - single_year_peak_share: max single-year share of lifetime issuances
      - months_since_last_issuance: staleness indicator

    Parameters
    ----------
    issuance_history : pd.DataFrame
        Must have columns: vintage_year, credits_issued

    Returns
    -------
    dict with computed metrics
    """
    if issuance_history.empty or "credits_issued" not in issuance_history.columns:
        return {
            "vintage_freshness_score": np.nan,
            "issuance_cv": np.nan,
            "single_year_peak_share": np.nan,
            "months_since_last_issuance": np.nan,
            "n_vintages": 0,
        }

    hist = issuance_history.copy()
    hist["credits_issued"] = pd.to_numeric(hist["credits_issued"], errors="coerce").fillna(0)
    hist = hist[hist["credits_issued"] > 0].copy()

    if hist.empty:
        return {
            "vintage_freshness_score": np.nan,
            "issuance_cv": np.nan,
            "single_year_peak_share": np.nan,
            "months_since_last_issuance": np.nan,
            "n_vintages": 0,
        }

    current_year = pd.Timestamp.now().year

    # Vintage freshness: weighted mean recency
    if "vintage_year" in hist.columns:
        hist["vintage_year"] = pd.to_numeric(hist["vintage_year"], errors="coerce")
        hist = hist.dropna(subset=["vintage_year"])
        total = hist["credits_issued"].sum()
        if total > 0:
            weighted_vintage = (hist["vintage_year"] * hist["credits_issued"]).sum() / total
            freshness = max(0, 1 - (current_year - weighted_vintage) / 10)
        else:
            freshness = np.nan
    else:
        freshness = np.nan

    # Issuance consistency (lower CV = more consistent = better)
    annual = hist.groupby("vintage_year")["credits_issued"].sum()
    cv = annual.std() / annual.mean() if annual.mean() > 0 else np.nan

    # Single-year peak share
    total_issued = annual.sum()
    peak_share = annual.max() / total_issued if total_issued > 0 else np.nan

    # Staleness
    if "issuance_date" in hist.columns:
        hist["issuance_date"] = pd.to_datetime(hist["issuance_date"], errors="coerce")
        last_date = hist["issuance_date"].max()
        if pd.notna(last_date):
            months_since = (pd.Timestamp.now() - last_date).days / 30.44
        else:
            months_since = np.nan
    elif "vintage_year" in hist.columns:
        last_vintage = hist["vintage_year"].max()
        months_since = (current_year - last_vintage) * 12
    else:
        months_since = np.nan

    return {
        "vintage_freshness_score": round(freshness, 4) if pd.notna(freshness) else np.nan,
        "issuance_cv": round(cv, 4) if pd.notna(cv) else np.nan,
        "single_year_peak_share": round(peak_share, 4) if pd.notna(peak_share) else np.nan,
        "months_since_last_issuance": round(months_since, 1) if pd.notna(months_since) else np.nan,
        "n_vintages": len(annual),
    }
