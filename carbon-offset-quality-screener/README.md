# Carbon Offset Quality Screener 🌿

A Python toolkit for evaluating the integrity and quality of voluntary carbon market (VCM) offset projects using publicly available registry data.

## Why This Exists

The voluntary carbon market is undergoing a credibility crisis. With the ICVCM Core Carbon Principles (CCPs) reshaping what counts as a high-integrity credit, buyers and analysts need reproducible, transparent tools to screen project quality before procurement or portfolio inclusion.

This toolkit pulls data directly from the **Verra VCS Registry** and scores each project on multiple integrity dimensions — permanence risk, additionality flags, vintage aging, issuance-to-retirement ratio, and project type risk profile — producing a composite quality index and portfolio-level risk summary.

## Features

- 📥 **Data ingestion** from Verra Registry public CSV export
- 🧮 **Composite Quality Index (CQI)** across 6 scoring dimensions
- 🚩 **Red flag detection** — automatically surfaces high-risk signals per project
- 📊 **Portfolio-level analytics** — concentration risk, vintage exposure, sector breakdown
- 📈 **Interactive visualizations** via Plotly
- 📄 **Exportable reports** in CSV and HTML

## Scoring Dimensions

| Dimension | Weight | Rationale |
|-----------|--------|-----------|
| Vintage quality | 20% | Credits >8 years old face increasing market discount |
| Issuance/Retirement ratio | 20% | Low retirement = low demand signal |
| Project type risk | 20% | REDD+/avoided deforestation historically higher controversy |
| Registry transparency | 15% | Completeness of public documentation |
| Additionality proxy | 15% | Registration timing relative to activity start |
| Geographical risk | 10% | Jurisdictional governance quality proxy |

## Installation

```bash
git clone https://github.com/Caro-rawr/carbon-offset-quality-screener.git
cd carbon-offset-quality-screener
pip install -r requirements.txt
```

## Quick Start

```python
from src.data_loader import VerraDataLoader
from src.scorer import CarbonQualityScorer
from src.visualizer import PortfolioVisualizer

loader = VerraDataLoader()
df = loader.load_projects(use_cache=True)

scorer = CarbonQualityScorer()
scored_df = scorer.score_all(df)

viz = PortfolioVisualizer(scored_df)
viz.quality_distribution()
viz.risk_heatmap()
```

```bash
python main.py --output reports/
```

## Data Source

[Verra Registry Projects Database](https://registry.verra.org) — public CSV, no API key required.

## Project Structure

```
carbon-offset-quality-screener/
├── src/
│   ├── data_loader.py
│   ├── scorer.py
│   ├── red_flags.py
│   └── visualizer.py
├── data/sample/
├── notebooks/01_quality_analysis_demo.ipynb
├── tests/test_scorer.py
├── reports/
└── main.py
```

## Author

Carolina Cruz Núñez | M.Sc. Sustainability Sciences 
[linkedin.com/in/carostrepto](https://linkedin.com/in/carostrepto)
