"""
data_loader.py
Ingests carbon offset project data from the Verra VCS Registry public database.
Falls back to sample data for offline development and testing.
"""

import os
import io
import logging
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Verra public CSV endpoint (VCS projects list)
VERRA_CSV_URL = (
    "https://registry.verra.org/uiapi/resource/resourceSummary/VCS?"
    "maxResults=2000&format=csv"
)

SAMPLE_DATA_PATH = Path(__file__).parent.parent / "data" / "sample" / "verra_sample.csv"
CACHE_PATH = Path(__file__).parent.parent / "data" / "raw" / "verra_cache.csv"


class VerraDataLoader:
    """
    Loads and standardizes project data from the Verra VCS Registry.
    
    Columns standardized:
        project_id, name, proponent, country, region, project_type,
        status, registration_date, crediting_period_start, crediting_period_end,
        total_issued, total_retired, total_buffer_pool, estimated_annual_reductions
    """

    REQUIRED_COLUMNS = [
        "project_id", "name", "country", "project_type",
        "status", "registration_date",
        "total_issued", "total_retired"
    ]

    # Mapping from Verra's raw column names to our standardized names
    COLUMN_MAP = {
        "ID": "project_id",
        "Name": "name",
        "Proponent": "proponent",
        "Country/Area": "country",
        "Region": "region",
        "Project Type": "project_type",
        "Status": "status",
        "Registration Date": "registration_date",
        "Crediting Period Start": "crediting_period_start",
        "Crediting Period End": "crediting_period_end",
        "Total Credits Issued": "total_issued",
        "Total Credits Retired": "total_retired",
        "Total Credits Cancelled": "total_cancelled",
        "Total Buffer Pool Credits": "total_buffer_pool",
        "Est. Annual GHG Reductions": "estimated_annual_reductions",
    }

    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_PATH.parent
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_projects(self, use_cache: bool = True, use_sample: bool = False) -> pd.DataFrame:
        """
        Main entry point. Loads project data with fallback priority:
          1. Sample data (if use_sample=True)
          2. Cached file (if use_cache=True and cache exists)
          3. Live download from Verra
          4. Sample data as last resort (offline fallback)
        
        Returns:
            pd.DataFrame with standardized columns
        """
        if use_sample:
            logger.info("Loading sample data (offline mode)")
            return self._load_sample()

        cache_file = self.cache_dir / "verra_cache.csv"

        if use_cache and cache_file.exists():
            logger.info(f"Loading from cache: {cache_file}")
            df = pd.read_csv(cache_file, low_memory=False)
            return self._clean_and_validate(df)

        logger.info("Attempting live download from Verra Registry...")
        df = self._download_verra()

        if df is not None:
            df.to_csv(cache_file, index=False)
            logger.info(f"Data cached to {cache_file}")
            return self._clean_and_validate(df)

        logger.warning("Download failed — falling back to sample data")
        return self._load_sample()

    def _download_verra(self) -> pd.DataFrame | None:
        """Downloads the Verra VCS projects list CSV."""
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (compatible; CarbonQualityScreener/1.0; "
                    "research use)"
                )
            }
            response = requests.get(VERRA_CSV_URL, headers=headers, timeout=30)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text), low_memory=False)
            logger.info(f"Downloaded {len(df)} projects from Verra")
            return self._rename_columns(df)
        except Exception as e:
            logger.error(f"Verra download failed: {e}")
            return None

    def _load_sample(self) -> pd.DataFrame:
        """Loads bundled sample data for offline testing."""
        if SAMPLE_DATA_PATH.exists():
            df = pd.read_csv(SAMPLE_DATA_PATH)
            logger.info(f"Loaded {len(df)} projects from sample data")
            return self._clean_and_validate(df)
        else:
            logger.warning("Sample file not found — generating synthetic data")
            return self._generate_synthetic_sample()

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies column name standardization."""
        return df.rename(columns={k: v for k, v in self.COLUMN_MAP.items() if k in df.columns})

    def _clean_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans data types, handles missing values, filters to active/registered projects.
        """
        df = df.copy()

        # Standardize column names if not already done
        df = self._rename_columns(df)

        # Numeric cleaning
        for col in ["total_issued", "total_retired", "total_buffer_pool",
                    "total_cancelled", "estimated_annual_reductions"]:
            if col in df.columns:
                df[col] = (
                    df[col].astype(str)
                    .str.replace(",", "")
                    .str.strip()
                    .replace("", "0")
                    .replace("nan", "0")
                )
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Date parsing
        for date_col in ["registration_date", "crediting_period_start", "crediting_period_end"]:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        # Filter to meaningful statuses
        if "status" in df.columns:
            keep_statuses = ["Registered", "Under Development", "Registration Requested"]
            df = df[df["status"].isin(keep_statuses) | df["status"].isna()]

        # Drop rows without a project ID
        if "project_id" in df.columns:
            df = df.dropna(subset=["project_id"])

        # Derived field: net credits (issued - retired)
        if "total_issued" in df.columns and "total_retired" in df.columns:
            df["net_credits"] = df["total_issued"] - df["total_retired"]

        df = df.reset_index(drop=True)
        logger.info(f"Cleaned dataset: {len(df)} projects")
        return df

    def _generate_synthetic_sample(self) -> pd.DataFrame:
        """
        Generates a synthetic sample dataset that mirrors Verra's structure.
        Used as final fallback for demos and tests.
        """
        import numpy as np
        np.random.seed(42)
        n = 200

        project_types = [
            "REDD+", "Improved Forest Management", "Afforestation/Reforestation",
            "Agriculture Forestry and Other Land Use",
            "Renewable Energy", "Energy Efficiency",
            "Methane Capture - Livestock", "Waste Handling and Disposal"
        ]
        countries = [
            "Brazil", "Indonesia", "Peru", "Colombia", "Mexico",
            "Kenya", "Tanzania", "Cambodia", "India", "China"
        ]
        regions = ["Latin America", "Asia Pacific", "Africa", "North America"]

        reg_dates = pd.date_range("2008-01-01", "2023-12-31", periods=n)

        total_issued = np.random.lognormal(mean=14, sigma=2, size=n).astype(int)
        retirement_rate = np.random.beta(2, 3, size=n)
        total_retired = (total_issued * retirement_rate).astype(int)

        df = pd.DataFrame({
            "project_id": [f"VCS{1000 + i}" for i in range(n)],
            "name": [f"Project {i}: Carbon Offset Initiative" for i in range(n)],
            "proponent": [f"Organisation {np.random.randint(1, 50)}" for _ in range(n)],
            "country": np.random.choice(countries, n),
            "region": np.random.choice(regions, n),
            "project_type": np.random.choice(project_types, n, p=[
                0.25, 0.10, 0.10, 0.10, 0.15, 0.10, 0.10, 0.10
            ]),
            "status": np.random.choice(
                ["Registered", "Under Development"], n, p=[0.85, 0.15]
            ),
            "registration_date": reg_dates,
            "crediting_period_start": reg_dates - pd.DateOffset(years=1),
            "crediting_period_end": reg_dates + pd.DateOffset(years=10),
            "total_issued": total_issued,
            "total_retired": total_retired,
            "total_buffer_pool": (total_issued * 0.1).astype(int),
            "total_cancelled": np.zeros(n, dtype=int),
            "estimated_annual_reductions": (total_issued / 10).astype(int),
        })

        df["net_credits"] = df["total_issued"] - df["total_retired"]

        # Save for reuse
        SAMPLE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(SAMPLE_DATA_PATH, index=False)
        logger.info(f"Synthetic sample saved to {SAMPLE_DATA_PATH}")
        return df
