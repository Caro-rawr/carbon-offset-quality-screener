"""
pipeline.py
-----------
End-to-end orchestration for the Carbon Offset Quality Screener.

Usage:
    python -m src.pipeline [OPTIONS]

Options:
    --project-type  TEXT    Filter by project type (e.g. "REDD+")
    --min-credits   INT     Minimum lifetime credits issued [default: 10000]
    --output-dir    PATH    Output directory [default: outputs/]
    --no-report             Skip HTML report generation
    --top-n         INT     Number of projects in report tables [default: 20]
"""

import logging
import argparse
import pandas as pd
from pathlib import Path

from src.fetcher import fetch_verra_projects
from src.cleaner import clean_project_data
from src.scorer import compute_quality_index, get_score_summary
from src.red_flags import detect_red_flags, get_flag_statistics
from src.visualizer import generate_html_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def run_pipeline(
    project_types: list = None,
    min_credits: int = 10_000,
    output_dir: Path = Path("outputs"),
    generate_report: bool = True,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Full screening pipeline: fetch → clean → score → flag → report.

    Returns
    -------
    pd.DataFrame
        Fully scored and flagged project dataset.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("CARBON OFFSET QUALITY SCREENER — Pipeline Start")
    logger.info("=" * 60)

    # --- Step 1: Fetch ---
    logger.info("Step 1/4: Fetching Verra registry data...")
    df_raw = fetch_verra_projects(
        project_types=project_types,
        min_credits_issued=min_credits,
        raw_data_dir=Path("data/raw"),
    )
    logger.info(f"  Retrieved {len(df_raw)} projects.")

    # --- Step 2: Clean ---
    logger.info("Step 2/4: Cleaning and standardizing data...")
    df_clean = clean_project_data(df_raw)
    logger.info(f"  Clean dataset: {len(df_clean)} projects, {len(df_clean.columns)} fields.")

    # --- Step 3: Score ---
    logger.info("Step 3/4: Computing Quality Index...")
    df_scored = compute_quality_index(df_clean)

    summary = get_score_summary(df_scored)
    logger.info(f"\n  Score dimension summary:\n{summary.to_string()}\n")

    # --- Step 4: Flag ---
    logger.info("Step 4/4: Detecting red flags...")
    df_flagged = detect_red_flags(df_scored)

    flag_stats = get_flag_statistics(df_flagged)
    logger.info(f"\n  Flag statistics:\n{flag_stats[['Flag', 'Severity', 'N Projects', '% of Total']].to_string(index=False)}\n")

    # --- Export CSVs ---
    full_csv = output_dir / "projects_screened_full.csv"
    df_flagged.to_csv(full_csv, index=False)
    logger.info(f"  Full dataset saved: {full_csv}")

    critical_csv = output_dir / "projects_critical_flags.csv"
    critical = df_flagged[df_flagged.get("max_severity", "NONE") == "CRITICAL"]
    if not critical.empty:
        critical.to_csv(critical_csv, index=False)
        logger.info(f"  Critical projects saved: {critical_csv} ({len(critical)} projects)")

    top_csv = output_dir / "projects_top_quality.csv"
    df_flagged.nlargest(top_n, "quality_index").to_csv(top_csv, index=False)
    logger.info(f"  Top {top_n} projects saved: {top_csv}")

    # --- HTML Report ---
    if generate_report:
        report_path = generate_html_report(
            df_flagged,
            output_path=output_dir / "screening_report.html",
            top_n=top_n,
        )
        logger.info(f"  HTML report: {report_path}")

    # --- Final summary ---
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Projects analyzed: {len(df_flagged)}")
    logger.info(f"  Mean QI: {df_flagged['quality_index'].mean():.1f}")
    logger.info(f"  High Quality: {(df_flagged['quality_tier'] == 'High Quality').sum()}")
    logger.info(f"  Medium Quality: {(df_flagged['quality_tier'] == 'Medium Quality').sum()}")
    logger.info(f"  Low Quality: {(df_flagged['quality_tier'] == 'Low Quality').sum()}")
    logger.info(f"  CRITICAL flags: {(df_flagged.get('max_severity', 'NONE') == 'CRITICAL').sum()}")
    logger.info("=" * 60)

    return df_flagged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Carbon Offset Quality Screener Pipeline")
    parser.add_argument("--project-type", type=str, default=None,
                        help="Project type filter, e.g. 'REDD+' or 'Renewable Energy'")
    parser.add_argument("--min-credits", type=int, default=10_000,
                        help="Minimum lifetime credits issued")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory path")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip HTML report generation")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of projects in summary tables")
    args = parser.parse_args()

    project_types = [args.project_type] if args.project_type else None

    run_pipeline(
        project_types=project_types,
        min_credits=args.min_credits,
        output_dir=Path(args.output_dir),
        generate_report=not args.no_report,
        top_n=args.top_n,
    )
