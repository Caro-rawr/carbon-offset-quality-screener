"""
visualizer.py
Plotly-based visualizations for carbon offset quality analysis.
All charts return plotly Figure objects (show() or write_html() as needed).
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TIER_COLORS = {
    "Very High": "#1a7340",
    "High": "#52b788",
    "Medium": "#f4a261",
    "Low": "#e07b39",
    "Very Low": "#c1121f",
}


class PortfolioVisualizer:
    """
    Generates interactive Plotly charts from scored + flagged project data.
    
    Args:
        df: DataFrame output from CarbonQualityScorer.score_all() and
            RedFlagDetector.detect()
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def quality_distribution(self, show: bool = True) -> go.Figure:
        """
        Histogram of CQI scores colored by quality tier.
        """
        fig = px.histogram(
            self.df,
            x="cqi",
            color="quality_tier",
            color_discrete_map=TIER_COLORS,
            nbins=30,
            title="Distribution of Composite Quality Index (CQI)",
            labels={"cqi": "CQI Score (0–100)", "quality_tier": "Quality Tier"},
            template="plotly_white",
            category_orders={"quality_tier": list(TIER_COLORS.keys())}
        )
        fig.update_layout(
            xaxis_title="Composite Quality Index",
            yaxis_title="Number of Projects",
            legend_title="Quality Tier",
            bargap=0.1
        )
        if show:
            fig.show()
        return fig

    def risk_heatmap(self, show: bool = True) -> go.Figure:
        """
        Heatmap of average CQI by country × project type.
        Reveals concentration of risk by geography and methodology.
        """
        if "country" not in self.df.columns or "project_type" not in self.df.columns:
            logger.warning("country or project_type column missing — skipping heatmap")
            return go.Figure()

        pivot = (
            self.df.groupby(["country", "project_type"])["cqi"]
            .mean()
            .reset_index()
            .pivot(index="country", columns="project_type", values="cqi")
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale=[
                    [0.0, "#c1121f"],
                    [0.3, "#e07b39"],
                    [0.5, "#f4a261"],
                    [0.7, "#52b788"],
                    [1.0, "#1a7340"],
                ],
                colorbar=dict(title="Avg CQI"),
                zmin=0,
                zmax=100,
                hoverongaps=False,
            )
        )
        fig.update_layout(
            title="Average CQI by Country × Project Type",
            xaxis_title="Project Type",
            yaxis_title="Country",
            template="plotly_white",
            height=600,
        )
        if show:
            fig.show()
        return fig

    def score_radar(self, project_id: str, show: bool = True) -> go.Figure:
        """
        Radar chart for a single project showing all 6 scoring dimensions.
        """
        row = self.df[self.df["project_id"].astype(str) == str(project_id)]
        if row.empty:
            logger.warning(f"Project {project_id} not found")
            return go.Figure()

        row = row.iloc[0]
        dimensions = [
            "vintage_score", "retirement_ratio_score", "project_type_score",
            "transparency_score", "additionality_score", "governance_score"
        ]
        labels = [
            "Vintage", "Retirement Ratio", "Project Type",
            "Transparency", "Additionality", "Governance"
        ]
        values = [float(row.get(d, 0)) for d in dimensions]
        values_closed = values + [values[0]]  # close the polygon

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=labels + [labels[0]],
            fill="toself",
            name=str(row.get("project_id", project_id)),
            line_color="#52b788",
            fillcolor="rgba(82,183,136,0.3)"
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title=f"Quality Profile — {row.get('name', project_id)[:60]}",
            template="plotly_white",
            showlegend=False
        )
        if show:
            fig.show()
        return fig

    def flag_frequency(self, flag_summary: pd.DataFrame, show: bool = True) -> go.Figure:
        """
        Bar chart showing which flags are most frequent in the portfolio.
        Requires output of RedFlagDetector.get_flag_summary().
        """
        if flag_summary.empty:
            return go.Figure()

        severity_colors = {"high": "#c1121f", "medium": "#e07b39", "low": "#f4a261"}

        fig = px.bar(
            flag_summary.sort_values("project_count", ascending=True),
            x="project_count",
            y="label",
            color="severity",
            color_discrete_map=severity_colors,
            orientation="h",
            title="Red Flag Frequency Across Portfolio",
            labels={
                "project_count": "Number of Projects",
                "label": "Flag",
                "severity": "Severity"
            },
            template="plotly_white",
            text="pct_of_portfolio"
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(
            xaxis_title="Number of Projects",
            yaxis_title="",
            legend_title="Severity",
            height=max(300, len(flag_summary) * 50)
        )
        if show:
            fig.show()
        return fig

    def scatter_cqi_vs_issuance(self, show: bool = True) -> go.Figure:
        """
        Scatter plot: CQI score vs total credits issued, colored by quality tier.
        Reveals if large issuance projects cluster at low quality (inflated baseline risk).
        """
        df_plot = self.df[self.df["total_issued"] > 0].copy()
        df_plot["issued_log"] = np.log10(df_plot["total_issued"])

        fig = px.scatter(
            df_plot,
            x="issued_log",
            y="cqi",
            color="quality_tier",
            color_discrete_map=TIER_COLORS,
            hover_data=["project_id", "name", "country", "project_type", "total_issued"],
            title="CQI vs. Total Credits Issued (log scale)",
            labels={
                "issued_log": "Total Credits Issued (log₁₀ tCO₂e)",
                "cqi": "CQI Score",
                "quality_tier": "Tier"
            },
            template="plotly_white",
            opacity=0.75,
            category_orders={"quality_tier": list(TIER_COLORS.keys())}
        )
        fig.update_layout(
            xaxis_title="Total Credits Issued (log₁₀ tCO₂e)",
            yaxis_title="Composite Quality Index",
        )
        if show:
            fig.show()
        return fig

    def portfolio_summary_card(self) -> dict:
        """
        Returns a dictionary with key portfolio metrics for dashboards or reports.
        """
        df = self.df
        return {
            "total_projects": len(df),
            "avg_cqi": round(df["cqi"].mean(), 1) if "cqi" in df.columns else None,
            "pct_high_quality": round((df["cqi"] >= 70).mean() * 100, 1) if "cqi" in df.columns else None,
            "pct_flagged": round((df.get("flag_count", pd.Series([0] * len(df))) > 0).mean() * 100, 1),
            "top_country": df["country"].value_counts().index[0] if "country" in df.columns else None,
            "top_project_type": df["project_type"].value_counts().index[0] if "project_type" in df.columns else None,
            "total_issued_mtco2": round(df["total_issued"].sum() / 1e6, 2) if "total_issued" in df.columns else None,
            "total_retired_mtco2": round(df["total_retired"].sum() / 1e6, 2) if "total_retired" in df.columns else None,
        }

    def save_all(self, output_dir: str) -> None:
        """Saves all charts as HTML files to the specified directory."""
        from pathlib import Path
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        self.quality_distribution(show=False).write_html(out / "1_quality_distribution.html")
        self.risk_heatmap(show=False).write_html(out / "2_risk_heatmap.html")
        self.scatter_cqi_vs_issuance(show=False).write_html(out / "3_scatter_cqi_issuance.html")
        logger.info(f"Charts saved to {out}")
