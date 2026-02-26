"""
main.py
CLI entry point for the Carbon Offset Quality Screener pipeline.

Usage:
    python main.py
    python main.py --sample          # use bundled sample data (offline)
    python main.py --output reports/ # specify output directory
    python main.py --top 20          # show top 20 projects by CQI
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.data_loader import VerraDataLoader
from src.scorer import CarbonQualityScorer
from src.red_flags import RedFlagDetector
from src.visualizer import PortfolioVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Carbon Offset Quality Screener — Verra VCS Registry Analysis"
    )
    parser.add_argument("--sample", action="store_true",
                        help="Use bundled sample data instead of downloading")
    parser.add_argument("--output", type=str, default="reports",
                        help="Output directory for CSV and HTML reports")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of top projects to display in summary")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force fresh download from Verra Registry")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ────────────────────────────────────────────────────
    logger.info("Step 1/4 — Loading project data")
    loader = VerraDataLoader()
    df = loader.load_projects(
        use_sample=args.sample,
        use_cache=not args.no_cache
    )
    logger.info(f"  Loaded {len(df):,} projects")

    # ── 2. Score ────────────────────────────────────────────────────────
    logger.info("Step 2/4 — Computing Composite Quality Index")
    scorer = CarbonQualityScorer()
    df = scorer.score_all(df)

    # ── 3. Flag ─────────────────────────────────────────────────────────
    logger.info("Step 3/4 — Running red flag detection")
    detector = RedFlagDetector()
    df = detector.detect(df)
    flag_summary = detector.get_flag_summary(df)

    # ── 4. Output ────────────────────────────────────────────────────────
    logger.info("Step 4/4 — Generating outputs")

    # CSV export
    csv_path = output_dir / "scored_projects.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"  Scored data saved to {csv_path}")

    flag_csv = output_dir / "flag_summary.csv"
    flag_summary.to_csv(flag_csv, index=False)
    logger.info(f"  Flag summary saved to {flag_csv}")

    # Charts
    viz = PortfolioVisualizer(df)
    viz.save_all(str(output_dir))

    # Console summary
    summary = viz.portfolio_summary_card()
    print("\n" + "═" * 55)
    print("  PORTFOLIO QUALITY SUMMARY")
    print("═" * 55)
    print(f"  Total projects analysed:  {summary['total_projects']:,}")
    print(f"  Average CQI:              {summary['avg_cqi']}")
    print(f"  High-quality projects:    {summary['pct_high_quality']}% (CQI ≥ 70)")
    print(f"  Projects with flags:      {summary['pct_flagged']}%")
    print(f"  Total issued (MtCO₂e):    {summary['total_issued_mtco2']:,.1f}")
    print(f"  Total retired (MtCO₂e):   {summary['total_retired_mtco2']:,.1f}")
    print("═" * 55)

    # Top N projects
    display_cols = ["project_id", "name", "country", "project_type", "cqi",
                    "quality_tier", "flag_count"]
    display_cols = [c for c in display_cols if c in df.columns]
    top_projects = df.nlargest(args.top, "cqi")[display_cols]

    print(f"\n  Top {args.top} projects by CQI:\n")
    print(top_projects.to_string(index=False))
    print()

    # Highest-risk projects
    worst = df.nsmallest(args.top, "cqi")[display_cols]
    print(f"  Bottom {args.top} projects by CQI (highest risk):\n")
    print(worst.to_string(index=False))
    print()

    logger.info(f"Pipeline complete. Reports saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
