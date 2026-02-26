"""
src/scraper.py
────────────────────────────────────────────────────────────────
Cliente para el Registro Público de Verra (VCS).

Verra expone una API REST pública en:
  https://registry.verra.org/uiapi/resource/resourceSummary/VCS

Esta API retorna proyectos de carbono con sus metadatos principales.
No requiere autenticación para consultas básicas de búsqueda.

Uso:
    from src.scraper import VerraRegistryScraper
    scraper = VerraRegistryScraper()
    projects = scraper.fetch_projects(n=50)          # últimos 50 proyectos
    projects_afolu = scraper.fetch_by_type("REDD+", n=30)
    df = scraper.to_dataframe(projects)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ─── Constantes de la API de Verra ───────────────────────────────────────────

VERRA_API_BASE = "https://registry.verra.org/uiapi"
VERRA_SEARCH_URL = f"{VERRA_API_BASE}/resource/resourceSummary/VCS"

# Campos que retorna la API y su mapeo a nombres internos limpios
FIELD_MAP = {
    "resourceIdentifier": "project_id",
    "resourceName": "project_name",
    "country": "country",
    "region": "region",
    "projectType": "project_type",
    "methodology": "methodology",
    "status": "status",
    "creditingPeriodStart": "crediting_start",
    "creditingPeriodEnd": "crediting_end",
    "estimatedAnnualEmissionReductions": "estimated_annual_er",
    "totalVCUsIssued": "credits_issued",
    "totalVCUsRetired": "credits_retired",
    "totalVCUsAvailable": "credits_available",
    "proponentName": "proponent",
    "validationVerificationBody": "verification_body",
    "registrationDate": "registration_date",
}


class VerraRegistryScraper:
    """Cliente para consultar el Registro Público de Verra VCS.

    Parameters
    ----------
    request_delay : float
        Segundos de espera entre requests para no saturar la API.
    timeout : int
        Timeout en segundos por request.
    cache_dir : Path, optional
        Directorio para cachear respuestas y evitar requests repetidos.
    """

    def __init__(
        self,
        request_delay: float = 1.0,
        timeout: int = 30,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.delay = request_delay
        self.timeout = timeout
        self.cache_dir = cache_dir
        self._session = self._build_session()

    def _build_session(self) -> requests.Session:
        """Construye sesión con reintentos automáticos."""
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.headers.update({
            "User-Agent": "carbon-offset-quality-screener/1.0 (research; carostrepto@gmail.com)",
            "Accept": "application/json",
        })
        return session

    # ─── Fetch métodos públicos ────────────────────────────────────────────

    def fetch_projects(
        self,
        n: int = 100,
        page_size: int = 50,
        status: str = "Registered",
    ) -> List[Dict[str, Any]]:
        """Obtiene proyectos VCS del registro público de Verra.

        Parameters
        ----------
        n : int
            Número máximo de proyectos a obtener.
        page_size : int
            Proyectos por página (máx. 50 en la API de Verra).
        status : str
            Filtro por estado: 'Registered', 'Under Validation', 'Rejected'.

        Returns
        -------
        list of dict
            Lista de proyectos con metadatos crudos.
        """
        all_projects: List[Dict[str, Any]] = []
        start_index = 0

        while len(all_projects) < n:
            batch_size = min(page_size, n - len(all_projects))
            params = {
                "maxResults": batch_size,
                "startIndex": start_index,
                "resourceStatus": status,
            }
            try:
                resp = self._session.get(
                    VERRA_SEARCH_URL, params=params, timeout=self.timeout
                )
                resp.raise_for_status()
                data = resp.json()

                # La API retorna {"totalCount": N, "documents": [...]}
                documents = data.get("documents", [])
                if not documents:
                    logger.info(f"No hay más proyectos disponibles (total obtenido: {len(all_projects)})")
                    break

                all_projects.extend(documents)
                start_index += len(documents)
                logger.debug(f"Obtenidos {len(all_projects)}/{n} proyectos")

                if len(documents) < batch_size:
                    break  # llegamos al final del registro

                time.sleep(self.delay)

            except requests.exceptions.RequestException as e:
                logger.error(f"Error en request a Verra API: {e}")
                break

        return all_projects[:n]

    def fetch_by_type(
        self,
        project_type: str,
        n: int = 50,
    ) -> List[Dict[str, Any]]:
        """Obtiene proyectos filtrados por tipo (REDD+, Renewable Energy, etc.).

        Parameters
        ----------
        project_type : str
            Tipo de proyecto a filtrar. La API de Verra acepta filtros
            en el campo 'resourceCategory'.
        n : int
            Número máximo de proyectos.
        """
        all_projects: List[Dict[str, Any]] = []
        start_index = 0

        while len(all_projects) < n:
            batch = min(50, n - len(all_projects))
            params = {
                "maxResults": batch,
                "startIndex": start_index,
                "resourceCategory": project_type,
                "resourceStatus": "Registered",
            }
            try:
                resp = self._session.get(
                    VERRA_SEARCH_URL, params=params, timeout=self.timeout
                )
                resp.raise_for_status()
                data = resp.json()
                documents = data.get("documents", [])
                if not documents:
                    break
                all_projects.extend(documents)
                start_index += len(documents)
                if len(documents) < batch:
                    break
                time.sleep(self.delay)
            except requests.exceptions.RequestException as e:
                logger.error(f"Error obteniendo proyectos tipo '{project_type}': {e}")
                break

        return all_projects[:n]

    def fetch_project_detail(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene el detalle completo de un proyecto por ID.

        Parameters
        ----------
        project_id : str
            ID del proyecto Verra (ej. 'VCS1234').
        """
        url = f"{VERRA_API_BASE}/resource/resourceSummary/VCS/{project_id}"
        try:
            resp = self._session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error obteniendo proyecto {project_id}: {e}")
            return None

    # ─── Transformación ────────────────────────────────────────────────────

    def clean_record(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Normaliza un registro crudo de la API a campos limpios.

        Parameters
        ----------
        raw : dict
            Registro crudo de la API de Verra.

        Returns
        -------
        dict
            Registro con campos renombrados y tipos corregidos.
        """
        clean: Dict[str, Any] = {}
        for api_field, clean_field in FIELD_MAP.items():
            clean[clean_field] = raw.get(api_field)

        # Normalizar numéricos
        for num_field in ["credits_issued", "credits_retired", "credits_available", "estimated_annual_er"]:
            val = clean.get(num_field)
            if val is not None:
                try:
                    clean[num_field] = float(str(val).replace(",", ""))
                except (ValueError, TypeError):
                    clean[num_field] = None

        # Calcular retirement_ratio
        issued = clean.get("credits_issued") or 0
        retired = clean.get("credits_retired") or 0
        clean["retirement_ratio"] = (retired / issued) if issued > 0 else None

        # Extraer año de vintage desde crediting_start
        cs = clean.get("crediting_start")
        if cs and isinstance(cs, str):
            try:
                clean["vintage_year"] = int(cs[:4])
            except (ValueError, IndexError):
                clean["vintage_year"] = None
        else:
            clean["vintage_year"] = None

        return clean

    def to_dataframe(self, projects: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convierte lista de proyectos (crudos o limpios) a DataFrame.

        Parameters
        ----------
        projects : list
            Lista de proyectos. Si son crudos de la API, se limpian automáticamente.

        Returns
        -------
        pd.DataFrame
        """
        if not projects:
            logger.warning("Lista de proyectos vacía.")
            return pd.DataFrame()

        # Detectar si son crudos (tienen 'resourceIdentifier') o ya limpios
        if "resourceIdentifier" in (projects[0] if projects else {}):
            records = [self.clean_record(p) for p in projects]
        else:
            records = projects

        df = pd.DataFrame(records)

        # Convertir fechas
        for date_col in ["registration_date", "crediting_start", "crediting_end"]:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        return df

    def save_raw(self, projects: List[Dict[str, Any]], path: Path) -> None:
        """Guarda proyectos crudos en JSON para reproducibilidad."""
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(projects, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Datos crudos guardados: {path} ({len(projects)} proyectos)")

    def save_processed(self, df: pd.DataFrame, path: Path) -> None:
        """Guarda DataFrame procesado en CSV."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8")
        logger.info(f"Datos procesados guardados: {path} ({len(df)} registros)")
