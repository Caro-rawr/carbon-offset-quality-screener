"""
fetcher.py
----------
Retrieves project and credit issuance data from the Verra VCS public registry.

Verra exposes a public REST API (no auth required) and downloadable CSVs.
This module handles both approaches with automatic fallback.

References:
  - Verra registry API: https://registry.verra.org/uiapi/resource/resourceSummary/VCS
  - Projects search: https://registry.verra.org/uiapi/resource/resourceSummary/VCS
"""

import time
import logging
import requests
import pandas as pd
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public Verra API endpoints (no key required)
# ---------------------------------------------------------------------------
VERRA_API_BASE = "https://registry.verra.org/uiapi"
PROJECTS_ENDPOINT = f"{VERRA_API_BASE}/resource/resourceSummary/VCS"
ISSUANCES_ENDPOINT = f"{VERRA_API_BASE}/resource/issuance/VCS"
RETIREMENTS_ENDPOINT = f"{VERRA_API_BASE}/resource/retirement/VCS"

# Fallback: Verra public CSV export (manually downloaded quarterly)
FALLBACK_CSV_URL = (
    "https://registry.verra.org/app/search/VCS/All%20Projects"
)

DEFAULT_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "carbon-offset-quality-screener/1.0 (research purposes)",
}

DEFAULT_PAGE_SIZE = 200
REQUEST_DELAY = 1.0  # seconds between paginated requests


def fetch_verra_projects(
    project_types: Optional[list] = None,
    min_credits_issued: int = 0,
    max_pages: int = 50,
    raw_data_dir: Path = Path("data/raw"),
) -> pd.DataFrame:
    """
    Download project-level summary data from Verra VCS registry.

    Parameters
    ----------
    project_types : list, optional
        Filter by project type labels, e.g. ['REDD+', 'Renewable Energy'].
        If None, all types are retrieved.
    min_credits_issued : int
        Minimum lifetime credits issued to include a project.
    max_pages : int
        Safety limit on API pagination.
    raw_data_dir : Path
        Where to cache raw JSON responses.

    Returns
    -------
    pd.DataFrame
        One row per VCS project with standardized column names.
    """
    raw_data_dir = Path(raw_data_dir)
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_data_dir / "verra_projects_raw.csv"

    # --- Try cached version first (avoid hammering the API) ---
    if cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < 24:
            logger.info(f"Loading from cache ({age_hours:.1f}h old): {cache_path}")
            df = pd.read_csv(cache_path, low_memory=False)
            return _apply_filters(df, project_types, min_credits_issued)

    # --- Live API fetch with pagination ---
    all_records = []
    page = 1

    while page <= max_pages:
        params = {
            "maxResults": DEFAULT_PAGE_SIZE,
            "startIndex": (page - 1) * DEFAULT_PAGE_SIZE,
            "isActive": "true",
        }

        try:
            resp = requests.get(
                PROJECTS_ENDPOINT,
                params=params,
                headers=DEFAULT_HEADERS,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.warning(f"API request failed on page {page}: {exc}")
            break

        records = data.get("value", data if isinstance(data, list) else [])
        if not records:
            break

        all_records.extend(records)
        logger.info(f"  → Fetched page {page} ({len(records)} projects)")

        # Verra returns fewer records than page_size on last page
        if len(records) < DEFAULT_PAGE_SIZE:
            break

        page += 1
        time.sleep(REQUEST_DELAY)

    if not all_records:
        logger.warning("No records retrieved from API. Using synthetic demo data.")
        return _load_synthetic_demo(raw_data_dir)

    df = pd.DataFrame(all_records)
    df = _standardize_project_columns(df)
    df.to_csv(cache_path, index=False)
    logger.info(f"Saved {len(df)} projects to {cache_path}")

    return _apply_filters(df, project_types, min_credits_issued)


def _standardize_project_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw Verra API field names to consistent internal schema.
    Field names may change across API versions; adjust mapping here.
    """
    column_map = {
        "resourceIdentifier": "project_id",
        "resourceName": "project_name",
        "country": "country",
        "region": "region",
        "projectType": "project_type",
        "methodologyIds": "methodologies",
        "totalVCUs": "credits_issued_total",
        "totalVCUsRetired": "credits_retired_total",
        "totalVCUsInBuffer": "credits_in_buffer",
        "registrationDate": "registration_date",
        "credentialingBody": "credentialing_body",
        "status": "status",
        "proponent": "proponent",
        "validators": "validators",
        "verifiers": "verifiers",
    }

    # Keep only columns that exist
    available = {k: v for k, v in column_map.items() if k in df.columns}
    df = df.rename(columns=available)

    # Numeric coercion
    for col in ["credits_issued_total", "credits_retired_total", "credits_in_buffer"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Parse dates
    if "registration_date" in df.columns:
        df["registration_date"] = pd.to_datetime(
            df["registration_date"], errors="coerce"
        )
        df["project_age_years"] = (
            pd.Timestamp.now() - df["registration_date"]
        ).dt.days / 365.25

    return df


def _apply_filters(
    df: pd.DataFrame,
    project_types: Optional[list],
    min_credits: int,
) -> pd.DataFrame:
    if project_types and "project_type" in df.columns:
        pattern = "|".join(project_types)
        df = df[df["project_type"].str.contains(pattern, case=False, na=False)]
    if min_credits > 0 and "credits_issued_total" in df.columns:
        df = df[df["credits_issued_total"] >= min_credits]
    logger.info(f"Returning {len(df)} projects after filters.")
    return df.reset_index(drop=True)


def fetch_issuance_history(
    project_id: str,
    raw_data_dir: Path = Path("data/raw"),
) -> pd.DataFrame:
    """
    Retrieve annual issuance time series for a single project.

    Parameters
    ----------
    project_id : str
        Verra project identifier (e.g. "VCS1234").

    Returns
    -------
    pd.DataFrame
        Columns: vintage_year, credits_issued, credits_retired
    """
    cache_path = Path(raw_data_dir) / f"issuances_{project_id}.csv"

    if cache_path.exists():
        return pd.read_csv(cache_path)

    try:
        resp = requests.get(
            f"{ISSUANCES_ENDPOINT}/{project_id}",
            headers=DEFAULT_HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        records = resp.json()
        df = pd.DataFrame(records)
        df = _standardize_issuance_columns(df)
        df.to_csv(cache_path, index=False)
        time.sleep(0.5)
        return df
    except requests.RequestException as exc:
        logger.warning(f"Could not fetch issuances for {project_id}: {exc}")
        return pd.DataFrame(columns=["vintage_year", "credits_issued", "credits_retired"])


def _standardize_issuance_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        "vintageYear": "vintage_year",
        "issuedCredits": "credits_issued",
        "retiredCredits": "credits_retired",
        "cancelledCredits": "credits_cancelled",
        "remainingCredits": "credits_remaining",
        "issuanceDate": "issuance_date",
    }
    available = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=available)
    for col in ["credits_issued", "credits_retired", "credits_cancelled", "credits_remaining"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def _load_synthetic_demo(raw_data_dir: Path) -> pd.DataFrame:
    """
    Generate synthetic but analytically realistic demo data when API is unavailable.
    All values are illustrative; structure matches the real Verra schema.
    """
    import numpy as np

    rng = np.random.default_rng(42)
    n = 80

    project_types = (
        ["REDD+"] * 20
        + ["Improved Forest Management"] * 10
        + ["Afforestation, Reforestation and Revegetation"] * 10
        + ["Renewable Energy"] * 20
        + ["Methane Capture"] * 10
        + ["Energy Efficiency"] * 10
    )
    rng.shuffle(project_types)

    issued = rng.integers(10_000, 5_000_000, size=n).astype(float)
    # Retirement rate varies: REDD+ lower, renewable higher
    base_retire_rate = [
        0.25 if "REDD" in t else
        0.40 if "Forest" in t else
        0.60 if "Renewable" in t else
        0.50
        for t in project_types
    ]
    noise = rng.uniform(-0.15, 0.15, size=n)
    retire_rate = np.clip(np.array(base_retire_rate) + noise, 0.01, 0.99)
    retired = (issued * retire_rate).astype(float)

    buffer_rate = rng.uniform(0.03, 0.25, size=n)
    buffer = (issued * buffer_rate).astype(float)

    reg_years = rng.integers(2010, 2023, size=n)
    registration_dates = pd.to_datetime(
        [f"{y}-{rng.integers(1,12):02d}-01" for y in reg_years]
    )

    df = pd.DataFrame(
        {
            "project_id": [f"VCS{1000 + i}" for i in range(n)],
            "project_name": [f"Demo Project {i+1} — {project_types[i][:20]}" for i in range(n)],
            "country": rng.choice(
                ["Brazil", "Peru", "Colombia", "Mexico", "Indonesia", "Kenya", "India", "China"],
                size=n,
            ),
            "project_type": project_types,
            "credits_issued_total": issued,
            "credits_retired_total": retired,
            "credits_in_buffer": buffer,
            "registration_date": registration_dates,
            "project_age_years": (pd.Timestamp.now() - registration_dates).days / 365.25,
            "verifiers": rng.choice(
                ["SCS Global", "DNV GL", "EY", "Bureau Veritas", "PwC"],
                size=n,
            ),
            "status": "Registered",
        }
    )

    out_path = raw_data_dir / "verra_projects_synthetic.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Synthetic demo data saved: {out_path}")
    return df
